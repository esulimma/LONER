#!/usr/bin/env python3

import argparse
import datetime
import os
import sys
from tqdm import tqdm
import csv

os.system("source /opt/ros/noetic/setup.bash")

import pandas as pd
import yaml
import rosbag
import torch

from pathlib import Path

import pickle
import copy
from typing import List

from sensor_msgs.msg import Image
from scipy.spatial.transform import Slerp

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))

sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")
sys.path.append(PROJECT_ROOT + "/analysis")
sys.path.append(PROJECT_ROOT + "/analysis/fdt_common_utils")

from mapping.optimizer import Optimizer

from src.common.sensors import Image, LidarScan
from src.common.ray_utils import LidarRayDirections

from src.common.settings import Settings

from src.common.pose_utils import compute_world_cube, WorldCube
from src.common.pose_utils import build_poses_from_df, dump_trajectory_to_tum

from src.models.losses import *
from src.models.model_tcnn import Model, OccupancyGridModel
from src.models.ray_sampling import OccGridRaySampler

from common.pose import Pose
from common.frame import Frame
from common.settings import Settings
from mapping.keyframe import KeyFrame

from examples.utils import *
from analysis.utils import *
from examples.fdt_optimize_implicit_map_utils import *
from analysis.fdt_common_utils.pcd_visualizer import *


################Settings#####################

CHUNK_SIZE = 2**8

ITERATE_LIDAR = True
LIDAR_MOTION_COMP = True
MAX_WINDOW_LENGTH_LIDAR = 16
N_EVAL = 6
SCHUFFLE = True
ITERATION_STRATEGY_LIDAR = 'MASK'
LIDAR_REPETITIONS_MAX = 8
SKIP_STEP_LIDAR = None
START_STEP_LIDAR = None
END_STEP_LIDAR = 120
L1_THRESHOLD = 1.05

NUM_ITERATIONS = 2**5

ITERATE_CAMERA = False
MAX_WINDOW_LENGTH_CAMERA = 6
VISULIZER = False
CAMERA_REPETITIONS = 1
IMAGE_UPSAMPLING = 2

IMAGE_MASK = False
ENLARGE_MASK = True
MASK_FROM_BAG = True
ITERATION_STRATEGY_CAMERA = 'FIXED'

SKIP_STEP_CAMERA = None
START_STEP_CAMERA = None
END_STEP_CAMERA = None


if IMAGE_UPSAMPLING is not None:
    IMAGE_UPSAMPLING = int(IMAGE_UPSAMPLING)

if START_STEP_LIDAR == None:
    START_STEP_LIDAR = 0

if END_STEP_LIDAR == None:
    END_STEP_LIDAR = -1

if SKIP_STEP_LIDAR == None:
    SKIP_STEP_LIDAR = 1


##################################################

def optimize_implicit_map(experiment_directory, configuration_path, ckpt_id=None, submap = None):
    """
    Optimize the implicit map using the given experiment directory and configuration path.

    Args:
        experiment_directory (str): The directory where the experiment files are located.
        configuration_path (str): The path to the configuration file.
        ckpt_id (str, optional): The checkpoint ID to load. If None, the latest checkpoint will be used.

    Returns:
        None
    """
    
    if experiment_directory is not None:
        """
        Load the configuration from the experiment directory.
        If their already exist a full config and a checkpoint, 
        Load them and reiterate the implicit map.
        """
        with open(f"{experiment_directory}/full_config.pkl", 'rb') as f:
            config = pickle.load(f)

        _log_directory = config['log_directory']

        ckpt_directory = _log_directory + "/checkpoints/"

        if ckpt_id is None:
            checkpoint_list = os.listdir(ckpt_directory)

            if len(checkpoint_list) == 0:
                print("No checkpoints found")
                return
            elif len(checkpoint_list) == 1:
                ckpt_id = checkpoint_list[0]
            else:
                max_iterations = 0

                for _ckpt in checkpoint_list:
                    if "final_" in _ckpt:
                        if float(_ckpt.split("final_")[1].split(".tar")[0]) > max_iterations:
                            max_iterations = float(_ckpt.split("final_")[1].split(".tar")[0])
                            ckpt_id = _ckpt
                    
                    if "reiterate_camera_" in _ckpt:
                        if float(_ckpt.split("reiterate_camera_")[1].split(".tar")[0]) > max_iterations:
                            max_iterations = float(_ckpt.split("reiterate_camera_")[1].split(".tar")[0])
                            ckpt_id = _ckpt
                            
                if max_iterations == 0:
                    ckpt_id = "final.tar"
            
        ckpt_path = ckpt_directory + ckpt_id
        ckpt = torch.load(ckpt_path)

        rosbag_path = Path(os.path.expanduser(config["dataset_path"]))

        calibration = None

    else:  
        """
        Iterate a new implicit map from the configuration path from scratch.
        """
        with open(configuration_path) as config_file:
            config = yaml.full_load(config_file)
        
        baseline_settings_path = os.path.expanduser(f"~/LonerSLAM/cfg/{config['baseline']}")

        settings = [Settings.load_from_file(baseline_settings_path)][0] 

        if config["changes"] is not None:
            settings.augment(config["changes"]) 
            im_scale_factor = settings.system.image_scale_factor

        rosbag_path = Path(os.path.expanduser(config["dataset"]))
        
        calibration = load_calibration(config["dataset_family"], config["calibration"])

        camera = settings.system.ros_names.camera
        lidar = settings.system.ros_names.lidar
        camera_suffix = settings.system.ros_names.camera_suffix
        topic_prefix = settings.system.ros_names.topic_prefix

        lidar_topic = f"{topic_prefix}/{lidar}"

        lidar_only = settings.system.lidar_only

        if calibration is not None:
            image_topic = f"{topic_prefix}/{camera}/{camera_suffix}"

            settings["calibration"] = calibration.to_dict(im_scale_factor)
            
            image_size = (settings.calibration.camera_intrinsic.height, settings.calibration.camera_intrinsic.width)
            lidar_to_camera = Pose.from_settings(settings.calibration.lidar_to_camera).get_transformation_matrix()
            camera_to_lidar = lidar_to_camera.inverse()

        else:
            camera_to_lidar = None
            image_size = None

        if config["groundtruth_traj"] is not None:
            ground_truth_file = os.path.expanduser(config["groundtruth_traj"])
            ground_truth_df = pd.read_csv(ground_truth_file, names=["timestamp","x","y","z","q_x","q_y","q_z","q_w"], delimiter=" ")
            if submap is not None:
                lidar_poses, timestamps = build_poses_from_df(ground_truth_df, False)
            else:
                lidar_poses, timestamps = build_poses_from_df(ground_truth_df, True)
            _, timestamps = build_buffer_from_poses(lidar_poses, timestamps)
        else:

            lidar_poses = None

        if settings.system.world_cube.compute_from_groundtruth:
            assert lidar_poses is not None, "Must provide groundtruth file, or set system.world_cube.compute_from_groundtruth=False"
            traj_bounding_box = None
        else:
            traj_bounding_box = settings.system.world_cube.trajectory_bounding_box

        ray_range = settings.mapper.optimizer.model_config.data.ray_range

        settings["experiment_name"] = config["experiment_name"]

        settings["run_config"] = config
            
        _world_cube = compute_world_cube(
        camera_to_lidar, settings.calibration.camera_intrinsic.k, image_size, lidar_poses, ray_range, padding=0.3, traj_bounding_box=traj_bounding_box, submap = submap)
        
        now = datetime.datetime.now()
        now_str = now.strftime("%m%d%y_%H%M%S")

        expname = settings.experiment_name    
        _experiment_name = f"{expname}_{now_str}"

        if submap is not None:
            _experiment_name = expname + '/'  + submap

        if _experiment_name is None:
            _log_directory = os.path.expanduser(f"{settings.system.log_dir_prefix}/{_experiment_name}/")
        else:
            _log_directory = os.path.expanduser(f"{settings.system.log_dir_prefix}/{_experiment_name}/")

        os.makedirs(_log_directory, exist_ok=True)

        ckpt_directory = _log_directory + "/checkpoints/"

        os.makedirs(ckpt_directory, exist_ok=True)

        os.makedirs(_log_directory + "/trajectory", exist_ok=True)

        _lidar_only = settings.system.lidar_only

        _dataset_path = Path(rosbag_path.as_posix()).resolve().as_posix()

        settings["experiment_name"] = _experiment_name
        settings["dataset_path"] = _dataset_path
        settings["log_directory"] = _log_directory
        settings["world_cube"] = {"scale_factor": _world_cube.scale_factor, "shift": _world_cube.shift}

        settings["mapper"]["experiment_name"] = _experiment_name
        settings["mapper"]["log_directory"] = _log_directory
        settings["mapper"]["lidar_only"] = _lidar_only

        settings["tracker"]["experiment_name"] = _experiment_name
        settings["tracker"]["log_directory"] = _log_directory
        settings["tracker"]["lidar_only"] = _lidar_only

        # Pass debug settings through
        for key in settings.debug.flags:
            settings["debug"][key] = settings.debug.flags[key] and settings.debug.global_enabled

        settings["mapper"]["debug"] = settings.debug
        settings["tracker"]["debug"] = settings.debug

        world_cube = _world_cube.as_dict()

        with open(f"{_log_directory}/world_cube.yaml", "w+") as f:
            yaml.dump(world_cube, f)

        with open(f"{_log_directory}/full_config.yaml", 'w+') as f:
            yaml.dump(settings, f)

        with open(f"{_log_directory}/full_config.pkl", 'wb+') as f:
            pickle.dump(settings, f)

        with open(f"{_log_directory}/full_config.pkl", 'rb') as f:
            config = pickle.load(f)

        ckpt = None

    ########################################## LOAD FULL CONFIG ##########################################  
    """
    Load and initizialize model, optimizer, world cube, and other settings from the configuration file.
    """
    _DEVICE = torch.device(config.mapper.device)
    scale_factor = config.world_cube.scale_factor.to(_DEVICE)
    shift = config.world_cube.shift

    world_cube = WorldCube(scale_factor, shift).to(_DEVICE)
    im_scale_factor = config.system.image_scale_factor
    ray_range = config.mapper.optimizer.model_config.data.ray_range
    enable_sky_segmentation = config.system.sky_segmentation

    if config["run_config"]["dataset_family"] == "fusion_portable":
        recompute_lidar_timestamps = True
        print("Warning: Re-computing the LiDAR timestamps. This should only be done on fusion portable.")
    else:
        recompute_lidar_timestamps = False

    if calibration is not None:
        settings["calibration"] = calibration.to_dict(im_scale_factor)
        image_size = (config.calibration.camera_intrinsic.height, config.calibration.camera_intrinsic.width)
        lidar_to_camera = Pose.from_settings(config.calibration.lidar_to_camera).get_transformation_matrix()
        camera_to_lidar = lidar_to_camera.inverse()

    else:
        camera_to_lidar = None
        image_size = None

    lidar_only = True

    optimizer = Optimizer(
        config.mapper.optimizer, calibration, world_cube, 0,
        config.debug.flags.use_groundtruth_poses,
        lidar_only,
        enable_sky_segmentation)

    optimizer._calibration = config.calibration
    optimizer._settings.debug = config.debug
    optimizer._settings.log_directory = config.log_directory

    lidar_to_camera = Pose.from_settings(optimizer._calibration.lidar_to_camera)
    
    model_config = config.mapper.optimizer.model_config.model
    model = Model(model_config).to(_DEVICE)

    if ckpt is not None:
        model.load_state_dict(ckpt['network_state_dict'])
        optimizer._model = model

    occ_model_config = config.mapper.optimizer.model_config.model.occ_model
    occ_model = OccupancyGridModel(occ_model_config).to(_DEVICE)

    if ckpt is not None:
        occ_model.load_state_dict(ckpt['occ_model_state_dict'])
        optimizer._occupancy_grid_model = occ_model

    optimizer._optimization_settings.freeze_poses = True
    optimizer._use_gt_poses = True
    optimizer._enable_sky_segmentation = True

    if ckpt is not None:
        optimizer._global_step = ckpt['global_step']


    source_bag_path = config['dataset_path']
    source_bag = rosbag.Bag(source_bag_path)
    lidar_topic = f"/{config.system.ros_names.lidar}"

    ################### LIDAR POSE INTERPOLATION ##########################################
    """
    Interpolate the LiDAR poses based on timestamps and groundtruth trajectory.
    """
    gt_traj_file = config.run_config.groundtruth_traj
    traj_gt = load_tum_trajectory(gt_traj_file)
    gt_ts_list= tumposes_ts_to_list(traj_gt)

    first_gt_tumpose = None
    key_quats = []
    key_t = []
    gt_poses = []

    for i, gt_pose in enumerate(traj_gt):

        if first_gt_tumpose == None:
            first_gt_tumpose = gt_pose
            first_gt_timestamp = gt_pose.timestamp

        if submap is not None:
            pose_inv = gt_pose.to_transform()
        else:
            pose_inv = np.linalg.inv(first_gt_tumpose.to_transform()) @ gt_pose.to_transform()


        timestamp = gt_ts_list[i]

        key_quats.append(R.from_matrix(pose_inv[:3,:3]).as_quat())
        key_t.append(pose_inv[:3,3])
        gt_poses.append(pose_inv)

    key_rots = R.from_quat(key_quats)

    gt_ts = np.array(gt_ts_list)-first_gt_timestamp

    slerp = Slerp(gt_ts, key_rots)

    lidar_ts_list = lidar_ts_to_seq(source_bag,lidar_topic,0)

    lidar_ts = np.array(lidar_ts_list) # + first_lidar_timestamp

    SCAN_TIME  =  np.round((lidar_ts[-1]-lidar_ts[0])/len(lidar_ts),2)

    lidar_ts_truncated = lidar_ts[(first_gt_timestamp <= lidar_ts -  SCAN_TIME) * (lidar_ts <= gt_ts_list[-1] )]
    
    lidar_ts_ = lidar_ts_truncated - first_gt_timestamp

    lidar_ts_motion_comp = lidar_ts_ - SCAN_TIME 

    translation_interpolated = np.array([np.interp(lidar_ts_, gt_ts, np.array(key_t)[:,0]), 
                            np.interp(lidar_ts_, gt_ts, np.array(key_t)[:,1]), 
                            np.interp(lidar_ts_, gt_ts, np.array(key_t)[:,2])]).T

    poses_interpolated = np.hstack((translation_interpolated, slerp(lidar_ts_).as_rotvec(degrees=False)))

    translation_interpolated_motion_comp = np.array([np.interp(lidar_ts_motion_comp, gt_ts, np.array(key_t)[:,0]), 
                    np.interp(lidar_ts_motion_comp, gt_ts, np.array(key_t)[:,1]), 
                    np.interp(lidar_ts_motion_comp, gt_ts, np.array(key_t)[:,2])]).T

    poses_interpolated_motion_comp = np.hstack((translation_interpolated_motion_comp, slerp(lidar_ts_motion_comp).as_rotvec(degrees=False)))

    np.random.seed(8)

    test_indices = np.random.choice(len(poses_interpolated[START_STEP_LIDAR:END_STEP_LIDAR]), N_EVAL, replace=False)
    train_indices = [i for i in range(len(poses_interpolated[START_STEP_LIDAR:END_STEP_LIDAR])) if i not in test_indices]

    test_poses = [poses_interpolated[START_STEP_LIDAR:END_STEP_LIDAR][i] for i in test_indices]
    train_poses = [poses_interpolated[START_STEP_LIDAR:END_STEP_LIDAR][i] for i in train_indices]

    test_poses_interpolated_motion_comp = [poses_interpolated_motion_comp[START_STEP_LIDAR:END_STEP_LIDAR][i] for i in test_indices]
    train_poses_interpolated_motion_comp = [poses_interpolated_motion_comp[START_STEP_LIDAR:END_STEP_LIDAR][i] for i in train_indices]


    if ckpt is None:
        lidar_poses=[]

        for i, pose in enumerate(train_poses):
            lidar_pose = {
                "timestamp": lidar_ts_[train_indices[i]],
                "lidar_to_camera": lidar_to_camera.get_pose_tensor().detach().cpu().clone(),
                "lidar_pose":  torch.from_numpy(pose),
                "gt_lidar_pose": torch.from_numpy(pose),
                "tracked_pose": torch.zeros(6),
            }
            lidar_poses.append(lidar_pose)

        keyframe_timestamps = torch.tensor([lidar_pose["timestamp"] for lidar_pose in lidar_poses])
        keyframe_trajectory = torch.stack([Pose(pose_tensor=lidar_pose["lidar_pose"]).get_transformation_matrix() for lidar_pose in lidar_poses])

        estimated_timestamps = torch.tensor([ts for ts in lidar_ts_])
        estimated_trajectory = torch.stack([Pose(pose_tensor=torch.from_numpy(pose_interpolated)).get_transformation_matrix() for pose_interpolated in poses_interpolated])

        dump_trajectory_to_tum(keyframe_trajectory, keyframe_timestamps, f"{_log_directory}/trajectory/keyframe_trajectory.txt")
        dump_trajectory_to_tum(estimated_trajectory, estimated_timestamps, f"{_log_directory}/trajectory/estimated_trajectory.txt")

        np.save(f"{_log_directory}/trajectory/first_gt_timestamp.npy", first_gt_timestamp)

    else:
        lidar_poses = ckpt["poses"]  
        keyframe_timestamps = torch.tensor([lidar_pose["timestamp"] for lidar_pose in lidar_poses])
        keyframe_trajectory = torch.stack([Pose(pose_tensor=lidar_pose["lidar_pose"]).get_transformation_matrix() for lidar_pose in lidar_poses])


    ################ IMPLICIT MAP EVALUATION SETUP ##########################################
    """
    Setup the evaluation of the implicit map using the test and evaluation poses.
    Test Poses are not included in the training data set.
    Evaluation Poses are included in the training data set.
    """

    test_jobs = []
    
    for test_pose, test_pose_motion_comp, test_i in zip(test_poses, test_poses_interpolated_motion_comp, test_indices):

        test_timestamp_ = lidar_ts_[test_i] + first_gt_timestamp

        test_pose = Pose(pose_tensor=torch.from_numpy(test_pose))
        initiliase_from_zero = False

        msg, timestamp =  get_msg_from_time_stamp(source_bag, lidar_topic, test_timestamp_, initiliase_from_zero)
        test_lidar_scan = build_scan_from_msg(msg, timestamp, recomute_timestamps = recompute_lidar_timestamps)

        if LIDAR_MOTION_COMP:
            test_pose_km1 = Pose(pose_tensor=torch.from_numpy(test_pose_motion_comp))
            test_lidar_scan.motion_compensate([test_pose_km1, test_pose], [0,SCAN_TIME], test_pose, use_gpu = True)

        test_ray_directions = LidarRayDirections(test_lidar_scan, CHUNK_SIZE)
        test_jobs.append((test_pose, test_ray_directions))

    eval_jobs_l1_depth = []
    eval_indices = np.random.choice(len(train_poses), N_EVAL, replace=False)


    for eval_i in eval_indices:

        eval_timestamp_ = keyframe_timestamps[eval_i].cpu().numpy() + first_gt_timestamp
        eval_pose = Pose(pose_tensor=torch.from_numpy(train_poses[eval_i]))
        initiliase_from_zero = False

        msg, timestamp =  get_msg_from_time_stamp(source_bag, lidar_topic, eval_timestamp_, initiliase_from_zero)
        eval_lidar_scan = build_scan_from_msg(msg, timestamp, recomute_timestamps = recompute_lidar_timestamps)

        if LIDAR_MOTION_COMP:
            eval_pose_km1 = Pose(pose_tensor=torch.from_numpy(train_poses_interpolated_motion_comp[eval_i]))
            eval_lidar_scan.motion_compensate([eval_pose_km1, eval_pose], [0,SCAN_TIME], eval_pose, use_gpu = True)

        eval_ray_directions = LidarRayDirections(eval_lidar_scan, CHUNK_SIZE)
        eval_jobs_l1_depth.append((eval_pose, eval_ray_directions))
        
    eval_timestamps = torch.tensor([ts for ts in keyframe_timestamps[eval_indices].cpu().numpy()])
    eval_trajectory = torch.stack([Pose(pose_tensor=torch.from_numpy(train_pose)).get_transformation_matrix() for train_pose in np.array(train_poses)[eval_indices]])

    test_timestamps = torch.tensor([ts for ts in lidar_ts_[test_indices]])
    test_trajectory = torch.stack([Pose(pose_tensor=torch.from_numpy(pose_interpolated)).get_transformation_matrix() for pose_interpolated in poses_interpolated[test_indices]])

    dump_trajectory_to_tum(eval_trajectory, eval_timestamps, f"{_log_directory}/trajectory/eval_trajectory.txt")
    dump_trajectory_to_tum(test_trajectory, test_timestamps, f"{_log_directory}/trajectory/test_trajectory.txt")

    ################ KEYFRAME SCHEDULE ##########################################
    """
    Set the keyframe schedule and configuration for the optimization
    of the sigma MLP.
    """
    optimizer._optimization_settings.num_iterations = NUM_ITERATIONS
    optimizer._keyframe_count = 1

    empty_kf_schedule = copy.deepcopy(optimizer._keyframe_schedule[0])
    empty_kf_schedule['num_keyframes'] = -1
    empty_kf_schedule['iteration_schedule'][0]['num_iterations'] = NUM_ITERATIONS
    empty_kf_schedule['iteration_schedule'][0]['freeze_poses'] = True
    empty_kf_schedule['iteration_schedule'][0]['freeze_sigma_mlp'] = False
    empty_kf_schedule['iteration_schedule'][0]['freeze_rgb_mlp'] = True
    
    kf_schedule = []
    kf_schedule.append(copy.deepcopy(empty_kf_schedule))
    optimizer._keyframe_schedule = kf_schedule

    window: List[KeyFrame] = []

    initiliase_from_zero = True
    l1s_mean_prev = float('inf')

    os.makedirs(f"{_log_directory}/metrics/", exist_ok=True)

    output_file = f"{_log_directory}/metrics/l1_test.csv"

    with open(output_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['global_step','min', 'max', 'mean', 'rmse'])
    csv_file.close()

    output_file = f"{_log_directory}/metrics/l1_eval.csv"

    with open(output_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['global_step','min', 'max', 'mean', 'rmse'])
    csv_file.close()

    ########################################## LIDAR ITERATION LOOP ##########################################
    """
    Run the iteration loop for the sigma MLP.
    """
    r = 0 
    
    if ITERATE_LIDAR:
        print("Starting Sigma MLP Optimization:")
        print(f"Using LiDAR Data from {lidar_topic}")

        optimizer._settings['rays_selection']['strategy'] = ITERATION_STRATEGY_LIDAR

        while r < LIDAR_REPETITIONS_MAX:
            if SCHUFFLE:
                schuffle_indices = np.random.choice(len(lidar_poses[::SKIP_STEP_LIDAR]), len(lidar_poses[::SKIP_STEP_LIDAR]), replace=False)
            else:
                schuffle_indices = np.arange(len(lidar_poses[::SKIP_STEP_LIDAR]))

            for i in schuffle_indices:

                lidar_pose = lidar_poses[::SKIP_STEP_LIDAR][i]
                timestamp_ = lidar_pose['timestamp'] + first_gt_timestamp
                initiliase_from_zero = False

                msg, timestamp =  get_msg_from_time_stamp(source_bag, lidar_topic, timestamp_, initiliase_from_zero)
                lidar_scan = build_scan_from_msg(msg, timestamp, config.system.lidar_fov, recompute_lidar_timestamps)
        
                lidar_pose_k = lidar_pose['lidar_pose']
                pose_k_ = Pose(pose_tensor = lidar_pose_k)

                if LIDAR_MOTION_COMP:
                    lidar_pose_km1 = torch.from_numpy(train_poses_interpolated_motion_comp[::SKIP_STEP_LIDAR][i])
                    pose_km1_ = Pose(pose_tensor = lidar_pose_km1)
                    lidar_scan.motion_compensate([pose_km1_,pose_k_], [0,SCAN_TIME], pose_k_, use_gpu = True)

                new_frame = Frame(None, lidar_scan, lidar_to_camera)
                new_frame._lidar_pose = pose_k_
                new_frame._gt_lidar_pose = pose_k_
                new_frame._id = i

                if enable_sky_segmentation:
                    compute_sky_rays(new_frame)
                else:
                    new_frame.lidar_points.sky_rays = torch.tensor([])
                new_keyframe = KeyFrame(new_frame)
                window.append(new_keyframe)

                if len(window) == MAX_WINDOW_LENGTH_LIDAR:
                    optimizer.iterate_optimizer(window)
                    window: List[KeyFrame] = []

            if len(window) > 0:
                optimizer.iterate_optimizer(window)
                window: List[KeyFrame] = []

            new_ckpt = {'global_step': optimizer._global_step,
                    'network_state_dict': optimizer._model.state_dict(),
                    'optimizer_state_dict': optimizer._optimizer.state_dict(),
                    'poses': lidar_poses,
                    'occ_model_state_dict': optimizer._occupancy_grid_model.state_dict(),
                    'occ_optimizer_state_dict': optimizer._occupancy_grid_optimizer.state_dict()}
        
            l1s = []

            _ray_sampler = OccGridRaySampler()
            _model_config = config.mapper.optimizer.model_config.model
            _model = Model(_model_config).to(_DEVICE)

            _occupancy_grid = occ_model()
            _ray_sampler.update_occ_grid(_occupancy_grid.detach())
            
            _model = optimizer._model 

            model_data = (_model, _ray_sampler, world_cube, ray_range, _DEVICE)

            for lidar_pose, ray_directions in tqdm(test_jobs):
                l1 = compute_l1_depth(lidar_pose, ray_directions, model_data, False)
                l1s.append(l1)

            l1s = torch.hstack(l1s)

            results_dir = f"{_log_directory}/metrics_test/"
            os.makedirs(results_dir, exist_ok=True)

            with open(f"{results_dir}/l1_{optimizer._global_step}.yaml", 'w+') as f:
                f.write(f"min: {l1s.min()}\nmax: {l1s.max()}\nmean: {l1s.mean()}\nrmse: {torch.sqrt(torch.mean(l1s**2))}")

            output_file = f"{_log_directory}/metrics/l1_test.csv"

            with open(output_file, 'a') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([optimizer._global_step, l1s.min().cpu().numpy(), l1s.max().cpu().numpy(), l1s.mean().cpu().numpy(), torch.sqrt(torch.mean(l1s**2)).cpu().numpy()])
            csv_file.close()

            l1s = []

            for lidar_pose, ray_directions in tqdm(eval_jobs_l1_depth):
                l1 = compute_l1_depth(lidar_pose, ray_directions, model_data, False)
                l1s.append(l1)

            l1s = torch.hstack(l1s)

            results_dir = f"{_log_directory}/metrics_eval/"
            os.makedirs(results_dir, exist_ok=True)

            with open(f"{results_dir}/l1_{optimizer._global_step}.yaml", 'w+') as f:
                f.write(f"min: {l1s.min()}\nmax: {l1s.max()}\nmean: {l1s.mean()}\nrmse: {torch.sqrt(torch.mean(l1s**2))}")

            output_file = f"{_log_directory}/metrics/l1_eval.csv"
            
            with open(output_file, 'a') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([optimizer._global_step, l1s.min().cpu().numpy(), l1s.max().cpu().numpy(), l1s.mean().cpu().numpy(), torch.sqrt(torch.mean(l1s**2)).cpu().numpy()])
            csv_file.close()

            r = r + 1

            if ckpt is None:
                ckpt_path = f"{config['log_directory']}/checkpoints/final.tar"
                torch.save(new_ckpt, ckpt_path)
                ckpt = new_ckpt

                print(f"Optimization Global Step: {optimizer._global_step}")
                print("Saved Checkpoint:")
                print(f"{ckpt_path}")
                if r == LIDAR_REPETITIONS_MAX:
                    print(f"Maximum repetitions: {LIDAR_REPETITIONS_MAX} reached!")
                    print("Sigma optimization is done!")
                elif l1s.mean() < L1_THRESHOLD:
                    print(f"L1 threshold: {L1_THRESHOLD} reached!")
                    print("Sigma optimization is done!")                
                if l1s.mean() < L1_THRESHOLD:
                    r = LIDAR_REPETITIONS_MAX
                    print(f"L1 threshold: {L1_THRESHOLD} reached!")
                    print("Sigma optimization is done!")

            else:
                ckpt_path = f"{config['log_directory']}/checkpoints/final_{optimizer._global_step}.tar"

                if r == LIDAR_REPETITIONS_MAX:
                    torch.save(new_ckpt, ckpt_path)
                    ckpt = new_ckpt
                    print(f"Maximum repetitions: {LIDAR_REPETITIONS_MAX} reached!")
                    print(f"Checkpoint saved: {ckpt_path}")
                    print("Sigma optimization is done!")

                if l1s.mean() < L1_THRESHOLD:
                    r = LIDAR_REPETITIONS_MAX
                    torch.save(new_ckpt, ckpt_path)
                    ckpt = new_ckpt
                    print(f"L1 threshold: {L1_THRESHOLD} reached!")
                    print(f"Checkpoint saved: {ckpt_path}")
                    print("Sigma optimization is done!")

                elif l1s_mean_prev < l1s.mean():            
                    r = LIDAR_REPETITIONS_MAX
                    print(f"L1 score: {l1s.mean()} lower than previous iteration {l1s_mean_prev}")
                    print("Sigma optimization is done!")

                else: 
                    l1s_mean_prev = l1s.mean()
                    torch.save(new_ckpt, ckpt_path)
                    ckpt = new_ckpt
                    print(f"Checkpoint saved: {ckpt_path}")


    ################ RGB-MLP Iteration ##########################################
    """
    Set the keyframe schedule, pose interpolation and configuration for the optimization
    of the RGB MLP.
    """

    optimizer._use_gt_poses = True
    optimizer._optimization_settings.freeze_poses = True
    optimizer._optimization_settings.freeze_sigma_mlp = True 
    optimizer._optimization_settings.freeze_rgb_mlp = False
    optimizer._optimization_settings.lidar_only = False
    optimizer._enable_sky_segmentation = False

    optimizer._settings['rays_selection']['strategy'] = 'FIXED'

    optimizer._optimization_settings.num_iterations = NUM_ITERATIONS
    optimizer._keyframe_count = 1

    empty_kf_schedule = copy.deepcopy(optimizer._keyframe_schedule[0])
    empty_kf_schedule['num_keyframes'] = -1
    empty_kf_schedule['iteration_schedule'][0]['num_iterations'] = NUM_ITERATIONS
    empty_kf_schedule['iteration_schedule'][0]['freeze_poses'] = True
    empty_kf_schedule['iteration_schedule'][0]['freeze_sigma_mlp'] = True
    empty_kf_schedule['iteration_schedule'][0]['freeze_rgb_mlp'] = False
    
    kf_schedule = []
    kf_schedule.append(copy.deepcopy(empty_kf_schedule))
    optimizer._keyframe_schedule = kf_schedule

    source_bag_path = config['dataset_path']
    source_bag_path = source_bag_path.replace("_adjusted","")
    source_bag = rosbag.Bag(source_bag_path)


    image_topic = f"{config.system.ros_names.camera}{config.system.ros_names.camera_suffix}"

    camera_ts_list = lidar_ts_to_seq(source_bag, image_topic,0)

    camera_ts = np.array(camera_ts_list)

    camera_ts_truncated = camera_ts[(first_gt_timestamp <= camera_ts) * (camera_ts <= gt_ts_list[-1] - SCAN_TIME)]
    
    camera_ts_ = camera_ts_truncated - first_gt_timestamp

    camera_translation_interpolated = np.array([np.interp(camera_ts_, gt_ts, np.array(key_t)[:,0]), 
                            np.interp(camera_ts_, gt_ts, np.array(key_t)[:,1]), 
                            np.interp(camera_ts_, gt_ts, np.array(key_t)[:,2])]).T

    camera_poses_interpolated = np.hstack((camera_translation_interpolated, slerp(camera_ts_).as_rotvec(degrees=False)))

    camera_poses_interpolated = camera_poses_interpolated[START_STEP_CAMERA:END_STEP_CAMERA:SKIP_STEP_CAMERA,:]
    camera_ts_ = camera_ts_[START_STEP_CAMERA:END_STEP_CAMERA:SKIP_STEP_CAMERA]

    camera_trajectory = torch.stack([Pose(pose_tensor=torch.from_numpy(camera_pose)).get_transformation_matrix() for camera_pose in camera_poses_interpolated])

    dump_trajectory_to_tum(camera_trajectory, torch.tensor([ts for ts in camera_ts_]), f"{_log_directory}/trajectory/camera_trajectory.txt")

    initiliase_from_zero = False
    empyt_lidar_scan = LidarScan()

    config["calibration"]["camera_intrinsic"]["k"] = torch.tensor(config["calibration"]["camera_intrinsic"]["k"])
    config["calibration"]["camera_intrinsic"]["new_k"] = config["calibration"]["camera_intrinsic"]["k"]
    config["calibration"]["camera_intrinsic"]["distortion"] = torch.tensor(config["calibration"]["camera_intrinsic"]["distortion"])
    config["calibration"]["lidar_to_camera"]["orientation"] = np.array(config["calibration"]["lidar_to_camera"]["orientation"]) # for weird compatability 
    config["calibration"]["lidar_to_camera"]["xyz"] = np.array(config["calibration"]["lidar_to_camera"]["xyz"])

    if (IMAGE_UPSAMPLING is not None):
        if (IMAGE_UPSAMPLING > 1):
            # https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix

            config["calibration"]["camera_intrinsic"]["width"] = int(config["calibration"]["camera_intrinsic"]["width"]  * IMAGE_UPSAMPLING)
            config["calibration"]["camera_intrinsic"]["height"] = int(config["calibration"]["camera_intrinsic"]["height"]  * IMAGE_UPSAMPLING)
            k_adjusted = config["calibration"]["camera_intrinsic"]["k"]
            k_adjusted[0,:] = k_adjusted[0,:] * (IMAGE_UPSAMPLING)
            k_adjusted[1,:] = k_adjusted[1,:] * (IMAGE_UPSAMPLING)
    
    optimizer._calibration = config.calibration

    mask_topic = f"{config.system.ros_names.camera}mask"

    if not MASK_FROM_BAG and IMAGE_MASK:
        mask_path = os.path.dirname(source_bag_path) + "/accumulated_mask.npy"

        mask = np.load(mask_path)

        if ENLARGE_MASK:
            mask = enlarge_mask(mask, 2.0)

        if (IMAGE_UPSAMPLING is not None) & (IMAGE_UPSAMPLING > 1):
            mask = upsample_mask(mask, IMAGE_UPSAMPLING)

    repetitions = CAMERA_REPETITIONS
    r = 0

    if ITERATE_CAMERA == True:
        print("Starting RGB MLP Optimization:")
        print(f"Using Camera Data from {image_topic}")
        if MASK_FROM_BAG:
            print(f"Using Dynamic Mask Data from {mask_topic}")
        elif IMAGE_MASK:
            print(f"Using Static Mask Data from {mask_path}")
        else:
            print("Not using any mask data")

        while r < repetitions:
            if SCHUFFLE:
                schuffle_indices = np.random.choice(len(camera_poses_interpolated), len(camera_poses_interpolated), replace=False)
            else:
                schuffle_indices = np.arange(len(camera_poses_interpolated))

            for i in schuffle_indices:

                timestamp_ = camera_ts_[i] + first_gt_timestamp

                msg, timestamp =  get_msg_from_time_stamp(source_bag, image_topic, timestamp_, initiliase_from_zero)
                camera_image = build_image_from_compressed_msg(msg, timestamp, IMAGE_UPSAMPLING)
                
                camera_T = camera_trajectory[i]             
                
                new_frame = Frame(camera_image, empyt_lidar_scan , lidar_to_camera)
                new_frame._lidar_pose = Pose(transformation_matrix = camera_T)
                new_frame._gt_lidar_pose = Pose(transformation_matrix = camera_T)
                new_frame._id = i

                if MASK_FROM_BAG:
                    mask_msg, timestamp =  get_msg_from_time_stamp(source_bag, mask_topic, timestamp_, initiliase_from_zero)
                    frame_mask = build_image_from_compressed_mask(mask_msg, timestamp, IMAGE_UPSAMPLING, ENLARGE_MASK)

                elif IMAGE_MASK:
                    frame_mask = Image(torch.from_numpy(mask), timestamp.to_sec())
                else:
                    frame_mask = None

                new_frame.mask = frame_mask
                new_keyframe = KeyFrame(new_frame)

                window.append(new_keyframe)

                if len(window) == MAX_WINDOW_LENGTH_CAMERA or i == schuffle_indices[-1]:
                    optimizer.iterate_optimizer_camera(window)
                    window: List[KeyFrame] = []

            if len(window) > 0:
                optimizer.iterate_optimizer_camera(window)

            r = r + 1

        ckpt = {'global_step': optimizer._global_step,
                'network_state_dict': optimizer._model.state_dict(),
                'optimizer_state_dict': optimizer._optimizer.state_dict(),
                'poses': lidar_poses,
                'occ_model_state_dict': optimizer._occupancy_grid_model.state_dict(),
                'occ_optimizer_state_dict': optimizer._occupancy_grid_optimizer.state_dict()}
        
        ckpt_save_name = f"reiterate_camera_{optimizer._global_step}.tar"
        
        torch.save(ckpt, f"{config['log_directory']}/checkpoints/" + ckpt_save_name)
        print("RGB-Optimization is done!")
        print(f"Checkpoint saved to folder: {config['log_directory']}")
        print(ckpt_save_name)

    config_lines = [
        f"CHUNK_SIZE: {CHUNK_SIZE}",
        f"ITERATE_LIDAR: {ITERATE_LIDAR}",
        f"LIDAR_MOTION_COMP: {LIDAR_MOTION_COMP}",
        f"MAX_WINDOW_LENGTH_LIDAR: {MAX_WINDOW_LENGTH_LIDAR}",
        f"N_EVAL: {N_EVAL}",
        f"SCHUFFLE: {SCHUFFLE}",
        f"ITERATION_STRATEGY_LIDAR: {ITERATION_STRATEGY_LIDAR}",
        f"LIDAR_REPETITIONS_MAX: {LIDAR_REPETITIONS_MAX}",
        f"SKIP_STEP_LIDAR: {SKIP_STEP_LIDAR}",
        f"START_STEP_LIDAR: {START_STEP_LIDAR}",
        f"END_STEP_LIDAR: {END_STEP_LIDAR}",
        f"L1_THRESHOLD: {L1_THRESHOLD}",
        f"NUM_ITERATIONS: {NUM_ITERATIONS}",
        f"ITERATE_CAMERA: {ITERATE_CAMERA}",
        f"MAX_WINDOW_LENGTH_CAMERA: {MAX_WINDOW_LENGTH_CAMERA}",
        f"VISULIZER: {VISULIZER}",
        f"CAMERA_REPETITIONS: {CAMERA_REPETITIONS}",
        f"IMAGE_UPSAMPLING: {IMAGE_UPSAMPLING}",
        f"IMAGE_MASK: {IMAGE_MASK}",
        f"ENLARGE_MASK: {ENLARGE_MASK}",
        f"MASK_FROM_BAG: {MASK_FROM_BAG}",
        f"ITERATION_STRATEGY_CAMERA: {ITERATION_STRATEGY_CAMERA}",
        f"SKIP_STEP_CAMERA: {SKIP_STEP_CAMERA}",
        f"START_STEP_CAMERA: {START_STEP_CAMERA}",
        f"END_STEP_CAMERA: {END_STEP_CAMERA}",
        f"",
        f"Experiment Name: {config['experiment_name']}",
        f"Configuration Path: {os.path.basename(configuration_path)}",
        f"Reoptimized from: {experiment_directory}",
        f"Checkpoint ID: {ckpt_id}",
        f"Bag: {config['dataset_path']}",
        f"Global Step: {optimizer._global_step}",
        f"Compute Capability: {torch.cuda.get_device_capability(0)}",
    ]

    if experiment_directory is not None:
        with open(f"{_log_directory}/optimizer_config.yaml", 'w+') as f:
            f.write("\n".join(config_lines))
    else:
        with open(f"{_log_directory}/optimizer_config_reiterize_{optimizer._global_step}.yaml", 'w+') as f:
            f.write("\n".join(config_lines))
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser("Optimizing Implicit Map")
    parser.add_argument("configuration_path")
    parser.add_argument("--experiment_directory", type=str, required=False, default=None, help="folder with outputs for reiteration")
    parser.add_argument("--gpu_ids", nargs="*", required=False, default = ["0"], help="Which GPUs to use")

    args = parser.parse_args()

    gpu_id = str(args.gpu_ids[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    optimize_implicit_map(args.experiment_directory, args.configuration_path)
