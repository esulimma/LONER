import open3d as o3d
import os 
import numpy as np
import sys
import pandas as pd
import torch
import pickle
import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")
sys.path.append(PROJECT_ROOT + "/analysis/fdt_common_utils")

data_path = os.path.dirname(PROJECT_ROOT) + '/data'
output_path = PROJECT_ROOT + '/outputs'
analysis_path = PROJECT_ROOT + '/analysis/fdt_analysis'


from render_utils import *
from analysis.utils import *

from src.common.pose import Pose
from src.common.ray_utils import CameraRayDirections, LidarRayDirections
from src.common.pose_utils import build_poses_from_df, WorldCube
from src.models.losses import *
from src.models.model_tcnn import Model, OccupancyGridModel
from src.models.ray_sampling import UniformRaySampler, OccGridRaySampler

from analysis.fdt_common_utils import build_lidar_scan
import fdt_render_dataset_frame
from examples.fdt_optimize_implicit_map_utils import *

SAVE_TENSORS = False
SAVE_MATPLOTS = False

CHUNK_SIZE = 2**12 
CHUNK_SIZE_LIDAR = 2**12
CHUNK_SIZE_CAMERA = 1024

def save_tensor_mplib_1(t1, filename, render_dir, titel1 = 'Variance', 
                        cmap = 'coolwarm'):
    fig, axes = plt.subplots(1, 1)
    im1 = axes[0].imshow(t1.squeeze().detach().cpu().numpy(), cmap='coolwarm')
    axes[0].set_title(titel1)
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    plt.tight_layout()
    # plt.show()
    plt.savefig(render_dir + '/' + filename)

def save_tensor_mplib_2(t1, t2, filename, render_dir, 
                        titel1 = 'Variance', title2 = 'Consistency', cmap = 'coolwarm'):
    fig, axes = plt.subplots(1, 2)
    im1 = axes[0].imshow(t1.squeeze().detach().cpu().numpy(), cmap='coolwarm')
    axes[0].set_title(titel1)
    axes[0].axis('off')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(t2.squeeze().detach().cpu().numpy(), cmap='coolwarm')
    axes[1].set_title(title2)
    axes[1].axis('off')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    # plt.show()
    plt.savefig(render_dir + '/' + filename)

def render_camera_trajectory(gt_ts_list_camera,traj_gt_camera, first_gt_timestamp,
                            img_size, render_path, lidar_to_camera,
                            model_data, camera_ray_directions):
    
    print(f"Rendering {len(traj_gt_camera)} camera frames")
    print("")

    render_path_RGB = render_path + "/RGB_renders"
    render_path_depth = render_path + "/depth_renders"
    os.makedirs(render_path_RGB, exist_ok=True)
    os.makedirs(render_path_depth, exist_ok=True)


    _, _, _, _, _DEVICE, _ = model_data

    for timestamp, gt_pose in tqdm.tqdm(zip(gt_ts_list_camera,traj_gt_camera), 
                                            desc="Rendering RGBD images"):


        adjusted_timestamp = first_gt_timestamp + timestamp.numpy()
        secs = int(adjusted_timestamp)
        nsecs = int(round(adjusted_timestamp - int(adjusted_timestamp),9) * 1e9)
        n_str = str(nsecs)

        if len(n_str) < 9:
            n_str = n_str.zfill(9)

        name = f"/{str(secs)}_" + n_str + ".png"

        lidar_pose = Pose(torch.tensor(gt_pose))
        cam_pose = lidar_pose.to('cpu') * lidar_to_camera.to('cpu')


        rgb_fine, depth_fine, peak_depth_consistency, peaks, variance, _ = fdt_render_dataset_frame.RGBD(
            cam_pose.to(_DEVICE), img_size, camera_ray_directions, model_data)

        save_depth(depth_fine,  name , render_path_depth, max_depth=75)
        save_rgb_cv2(rgb_fine, name , render_path_RGB)

        if SAVE_TENSORS:
            render_path_RGB_t = render_path + "/RGB_tensors"
            render_path_depth_t = render_path + "/depth_tensors"
            render_path_var_t = render_path + "/depth_var_tensors"
            render_path_con_t = render_path + "/depth_con_tensors"

            os.makedirs(render_path_RGB_t, exist_ok=True)
            os.makedirs(render_path_depth_t, exist_ok=True) 
            os.makedirs(render_path_var_t, exist_ok=True)
            os.makedirs(render_path_con_t, exist_ok=True)
            torch.save(rgb_fine, render_path_RGB_t)
            torch.save(depth_fine, render_path_depth_t)
            torch.save(variance, render_path_var_t)
            torch.save(peak_depth_consistency, render_path_con_t)

        if SAVE_MATPLOTS:
            render_path_peaks_m = render_path + "/peaks_matplots"
            render_path_depth_m = render_path + "/depth_matplots"
            render_path_var_m = render_path + "/depth_var_matplots"
            render_path_con_m = render_path + "/depth_con_matplots"

            os.makedirs(render_path_peaks_m, exist_ok=True)
            os.makedirs(render_path_depth_m, exist_ok=True)
            os.makedirs(render_path_var_m, exist_ok=True)
            os.makedirs(render_path_con_m, exist_ok=True)

            save_tensor_mplib_1(peaks, name, render_path_peaks_m, titel1='Peaks')
            save_tensor_mplib_1(depth_fine, name, render_path_depth_m, titel1='Depth')
            save_tensor_mplib_1(variance, name, render_path_var_m, titel1='Variance')
            save_tensor_mplib_1(peak_depth_consistency, name, render_path_con_m, titel1='Consistency')

            render_path_depths_peaks_m = render_path + "/depths_peaks_matplots"
            render_path_var_con_m = render_path + "/var_con_matplots"
            save_tensor_mplib_2(depth_fine, peaks, name, render_path_depths_peaks_m, titel1='Depth', title2='Peaks')
            save_tensor_mplib_2(variance, peak_depth_consistency, name, render_path_var_con_m, titel1='Variance', title2='Peak-Depth Consistency')


def render_LiDAR_trajectory(gt_ts_list_lidar,traj_gt_lidar,first_gt_timestamp,
                            render_path, 
                            model_data, QT64_ray_directions, settings):
    
    print(f"Rendering {len(traj_gt_lidar)} LiDAR scans")
    print("")

    _, _, _, _, _DEVICE, _ = model_data

    render_path_lidar_pcd = render_path + "/pcd_renders"
    render_path_lidar_pcd_L1 = render_path + "/pcd_L1_renders"
    render_path_lidar_pcd_acc = render_path + "/pcd_renders_accumulated"

    os.makedirs(render_path_lidar_pcd, exist_ok=True)
    os.makedirs(render_path_lidar_pcd_L1, exist_ok=True)
    os.makedirs(render_path_lidar_pcd_acc, exist_ok=True)

    merged_pcd = o3d.geometry.PointCloud()

    # for timestamp, gt_lidar_pose in tqdm.tqdm(zip(gt_ts_list_lidar[0:2],traj_gt_lidar[0:2]), 
    for timestamp, gt_lidar_pose in tqdm.tqdm(zip(gt_ts_list_lidar,traj_gt_lidar), 
                                            desc="Rendering LiDAR scans"):

        adjusted_timestamp = first_gt_timestamp + timestamp.numpy()
        secs = int(adjusted_timestamp)
        nsecs = int(round(adjusted_timestamp - int(adjusted_timestamp),9) * 1e9)
        n_str = str(nsecs)
        if len(n_str) < 9:
            n_str = n_str.zfill(9)
        name = f"/{str(secs)}_" + n_str + ".pcd"

        
        lidar_pose= Pose(torch.tensor(gt_lidar_pose)).to(_DEVICE)

        rendered_lidar = fdt_render_dataset_frame.LiDAR(lidar_pose, QT64_ray_directions, 
                                                    model_data, threshold = None, render_strategy = settings.render_strategy)

        rendered_pcd = o3d.geometry.PointCloud()
        rendered_pcd.points = o3d.utility.Vector3dVector(rendered_lidar)

        o3d.io.write_point_cloud(render_path_lidar_pcd + name, rendered_pcd)

        voxel_size = 0.1

        merged_pcd = merge_o3d_pc(merged_pcd, rendered_pcd.voxel_down_sample(voxel_size).transform(lidar_pose.get_transformation_matrix().cpu().numpy()))

    o3d.io.write_point_cloud(render_path_lidar_pcd_acc + "/merged_cloud_rendered.pcd", merged_pcd)


def main(render_path, experiment_directory, settings):
    """
    Main function to render results from. Loads a checkpoint, initializes models and configurations,
    and renders LiDAR and camera trajectories based on the given settings.

    Args:
        render_path (str): The path where the rendered trajectories will be saved.
        experiment_directory (str): The directory containing the experiment data.
        settings (object): The settings object containing various rendering options.

    Returns:
        tuple: A tuple containing the model data used for rendering.

    """

    # Load checkpoint
    ckpt_id = settings.ckpt_id

    with open(f"{experiment_directory}/full_config.pkl", 'rb') as f:
        full_config = pickle.load(f)
        _log_directory = full_config['log_directory']
        ckpt_directory = _log_directory + "/checkpoints/"

        # If checkpoint ID is not provided, select the latest checkpoint
        if ckpt_id is None:
            checkpoint_list = os.listdir(ckpt_directory)

            if len(checkpoint_list) == 0:
                print("No checkpoints found")
                exit()

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
            
    print(f"Loading checkpoint {ckpt_id}")
    
    try:
        ckpt = torch.load(ckpt_directory + ckpt_id)
    except:
        ckpt = torch.load(experiment_directory + '/checkpoints/' + ckpt_id)

    # Initialize models and configurations
    full_config["dataset_path"] = full_config["dataset_path"].replace("-", "_")
    _DEVICE = torch.device(full_config.mapper.device)
    scale_factor = full_config.world_cube.scale_factor.to(_DEVICE)
    shift = full_config.world_cube.shift
    world_cube = WorldCube(scale_factor, shift).to(_DEVICE)

    model_config = full_config.mapper.optimizer.model_config.model
    model = Model(model_config).to(_DEVICE)
    model.load_state_dict(ckpt['network_state_dict'])

    if full_config.mapper.optimizer.samples_selection.strategy == 'OGM':
        occ_model_config = full_config.mapper.optimizer.model_config.model.occ_model
        assert isinstance(occ_model_config, dict), f"OGM enabled but model.occ_model is empty"
        occ_model = OccupancyGridModel(occ_model_config).to(_DEVICE)
        ray_sampler = OccGridRaySampler()
        occ_model.load_state_dict(ckpt['occ_model_state_dict'])

        occupancy_grid = occ_model()
        ray_sampler.update_occ_grid(occupancy_grid.detach())
    else:
        ray_sampler = UniformRaySampler()


    ray_range = (2.5,settings.max_Lidar_ray_range)

    full_config["calibration"]["camera_intrinsic"]["k"] = torch.tensor(full_config["calibration"]["camera_intrinsic"]["k"])
    full_config["calibration"]["camera_intrinsic"]["distortion"] = torch.tensor(full_config["calibration"]["camera_intrinsic"]["distortion"])
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if full_config.mapper.optimizer.samples_selection.strategy == 'OGM':
        occ_model_config = full_config.mapper.optimizer.model_config.model.occ_model
        assert isinstance(occ_model_config, dict), f"OGM enabled but model.occ_model is empty"
        occ_model = OccupancyGridModel(occ_model_config).to(_DEVICE)
        ray_sampler = OccGridRaySampler()
        occ_model.load_state_dict(ckpt['occ_model_state_dict'])
        occupancy_grid = occ_model()
        ray_sampler.update_occ_grid(occupancy_grid.detach())
    else:
        ray_sampler = UniformRaySampler()


    ########################################## LOAD LIDAR CONFIG ##########################################

    eval_traj_file = f"{experiment_directory}/trajectory/eval_trajectory.txt"
    test_traj_file = f"{experiment_directory}/trajectory/test_trajectory.txt"
    keyframe_traj_file = f"{experiment_directory}/trajectory/keyframe_trajectory.txt"
    estimated_traj_file = f"{experiment_directory}/trajectory/keyframe_trajectory.txt"

    lidar_df = pd.read_csv(keyframe_traj_file, header=None, delimiter=" ")
    traj_gt_lidar, gt_ts_list_lidar = build_poses_from_df(lidar_df, False)
    eval_lidar_df = pd.read_csv(eval_traj_file, header=None, delimiter=" ")
    traj_eval_lidar, gt_ts_list_eval_lidar = build_poses_from_df(eval_lidar_df, False)
    test_lidar_df = pd.read_csv(test_traj_file, header=None, delimiter=" ")
    traj_test_lidar, gt_ts_list_test_lidar = build_poses_from_df(test_lidar_df, False)

    full_df = pd.read_csv(estimated_traj_file, header=None, delimiter=" ")
    traj_gt_full, gt_ts_list_full = build_poses_from_df(full_df, False)

    first_gt_timestamp = np.load(f"{_log_directory}/trajectory/first_gt_timestamp.npy")

    model_data = (model, ray_sampler, world_cube, ray_range, _DEVICE, CHUNK_SIZE_LIDAR)

    QT64_lidar_scan = build_lidar_scan.QT64().to(_DEVICE)
    QT64_ray_directions = LidarRayDirections(QT64_lidar_scan, chunk_size = CHUNK_SIZE_LIDAR)

    if settings.render_LIDAR:
        if settings.render_traj_lidar == 'full':
            render_LiDAR_trajectory(gt_ts_list_full, traj_gt_full, first_gt_timestamp,
                                    render_path, model_data, QT64_ray_directions, settings)

        if settings.render_traj_lidar == 'keyframe':
            render_LiDAR_trajectory(gt_ts_list_lidar, traj_gt_lidar, first_gt_timestamp,
                                    render_path, model_data, QT64_ray_directions, settings)
            
        if settings.render_traj_lidar == 'test':
            render_LiDAR_trajectory(gt_ts_list_test_lidar, traj_test_lidar, first_gt_timestamp,
                                    render_path, model_data, QT64_ray_directions, settings)
            
        if settings.render_traj_lidar == 'eval':
            render_LiDAR_trajectory(gt_ts_list_eval_lidar, traj_eval_lidar, first_gt_timestamp,
                                    render_path, model_data, QT64_ray_directions, settings)

    camera_traj_file = f"{experiment_directory}/trajectory/camera_trajectory.txt"
    camera_df = pd.read_csv(camera_traj_file, header=None, delimiter=" ")
    traj_gt_camera, gt_ts_list_camera = build_poses_from_df(camera_df, False)

    intrinsic = full_config.calibration.camera_intrinsic
    im_size = torch.Tensor([intrinsic.height, intrinsic.width])

    model_data_camera = (model, ray_sampler, world_cube, ray_range, _DEVICE, CHUNK_SIZE_CAMERA)

    camera_ray_directions = CameraRayDirections(full_config.calibration, chunk_size = CHUNK_SIZE_CAMERA, 
                                                device = _DEVICE)
    
    lidar_to_camera = Pose.from_settings(full_config.calibration.lidar_to_camera)

    if settings.render_Camera:
        render_camera_trajectory(gt_ts_list_camera, traj_gt_camera, first_gt_timestamp, im_size, 
                                render_path, lidar_to_camera, model_data_camera, camera_ray_directions)

    return model_data
        