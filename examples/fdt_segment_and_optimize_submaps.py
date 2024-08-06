#!/usr/bin/env python3

import os
import sys
import yaml
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))

sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT + "/src")
sys.path.append(PROJECT_ROOT + "/analysis")
sys.path.append(PROJECT_ROOT + "/analysis/fdt_common_utils")

from src.common.settings import Settings
from examples.fdt_optimize_implicit_map import optimize_implicit_map
from examples.utils import *
from analysis.utils import *
from examples.fdt_optimize_implicit_map_utils import *
from analysis.fdt_common_utils.pcd_visualizer import *

MAX_LENGTH = 50 # in Meters
PADDING = 30 # Number of Poses

def calculate_distance(pose1, pose2):
    """Calculate the Euclidean distance between two poses."""
    translation1 = np.array([pose1.x, pose1.y, pose1.z])
    translation2 = np.array([pose2.x, pose2.y, pose2.z])
    return np.linalg.norm(translation1 - translation2)

def middle_point(pose1, pose2):
    """Calculate the middle point between two poses."""
    translation1 = np.array([pose1.x, pose1.y, pose1.z])
    translation2 = np.array([pose2.x, pose2.y, pose2.z])
    return (translation1 + translation2) / 2

def optimize_and_segment_implicit_map(configuration_path):
    """
    Segment and optimize the implicit map using the given experiment directory and configuration path.

    Args:
        configuration_path (str): The path to the configuration file.

    Returns:
        None
    """
    with open(configuration_path) as config_file:
        config = yaml.full_load(config_file)

    baseline_settings_path = os.path.expanduser(f"~/LonerSLAM/cfg/{config['baseline']}")

    settings = [Settings.load_from_file(baseline_settings_path)][0] 
    if config["changes"] is not None:
        settings.augment(config["changes"]) 

    settings["experiment_name"] = config["experiment_name"]

    settings["run_config"] = config

    _experiment_name = settings.experiment_name    

    if _experiment_name is None:
        _log_directory = os.path.expanduser(f"{settings.system.log_dir_prefix}/{_experiment_name}/")
    else:
        _log_directory = os.path.expanduser(f"{settings.system.log_dir_prefix}/{_experiment_name}/")

    os.makedirs(_log_directory, exist_ok=True)

    log_dir_global = _log_directory
    
    gt_traj_file = config['groundtruth_traj']
    traj_gt = load_tum_trajectory(gt_traj_file)
    gt_ts_list= tumposes_ts_to_list(traj_gt)

    # Split trajectory based on distance
    split_trajectories = []
    split_ts_lists = []
    current_part = []
    current_ts_list = []
    current_distance = 0.0

    previous_pose = None

    for i, gt_pose in enumerate(traj_gt):
        if previous_pose is None:
            previous_pose = gt_pose
            current_part.append(gt_pose)
            current_ts_list.append(gt_ts_list[i])
            continue

        distance = calculate_distance(previous_pose, gt_pose)
        if current_distance + distance > MAX_LENGTH:
            split_trajectories.append(current_part)
            split_ts_lists.append(current_ts_list)
            current_part = [previous_pose]  # Start new part with the previous pose
            current_ts_list = [gt_ts_list[i-1]]  # Include the timestamp of the previous pose
            current_distance = 0.0

        current_part.append(gt_pose)
        current_ts_list.append(gt_ts_list[i])
        current_distance += distance
        previous_pose = gt_pose

    # Append the last part
    if current_part:
        split_trajectories.append(current_part)
        split_ts_lists.append(current_ts_list)

    # Process each part and save the necessary files
    middle_points = []
    submap_strings = []

    num_parts = len(split_trajectories)

    for idx, (part, ts_list) in enumerate(zip(split_trajectories, split_ts_lists)):
    
        # Calculate middle point
        mid_pose = middle_point(part[0], part[-1])
        middle_points.append(mid_pose)
        
        # Create submap string
        submap_string = f"submap_{idx:03d}"
        submap_strings.append(submap_string)
        
        # Create directories if they don't exist
        trajectories_dir = os.path.join(log_dir_global, "trajectories")
        config_dir = os.path.join(log_dir_global, "config")
        os.makedirs(trajectories_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        
        # Save trajectory to CSV file
        traj_filename = os.path.join(trajectories_dir, f"{submap_string}.csv")
        with open(traj_filename, 'w') as f:
            if idx > 0:
                part_previous = split_trajectories[idx-1]
                ts_list_previous = split_ts_lists[idx-1]
                for i in range(-PADDING, -1):
                    f.write(f"{ts_list_previous[i]} {part_previous[i].x} {part_previous[i].y} {part_previous[i].z} {part_previous[i].qx} {part_previous[i].qy} {part_previous[i].qz} {part_previous[i].qw}\n")
            for j in range(len(ts_list)):
                f.write(f"{ts_list[j]} {part[j].x} {part[j].y} {part[j].z} {part[j].qx} {part[j].qy} {part[j].qz} {part[j].qw}\n")
            if idx < num_parts - 1:
                part_next = split_trajectories[idx+1]
                ts_list_next = split_ts_lists[idx+1]
                for k in range(1,PADDING):
                    f.write(f"{ts_list_next[k]} {part_next[k].x} {part_next[k].y} {part_next[k].z} {part_next[k].qx} {part_next[k].qy} {part_next[k].qz} {part_next[k].qw}\n")

            # Save config file
            config_filename = os.path.join(config_dir, f"{submap_string}.yaml")
            config['groundtruth_traj'] = traj_filename
            # with open(config_filename, 'w') as f:
                # yaml.dump(config, f)
            print(f"Optimizing submap {submap_string}")
            optimize_implicit_map(None, config_filename, submap = submap_string)

    # Save middle points and submap strings as .np file
    middle_points_array = np.array(middle_points)
    submap_strings_array = np.array(submap_strings)

    np.save(os.path.join(log_dir_global, "submaps_middlepoints.npy"),  middle_points_array)
    np.save(os.path.join(log_dir_global, "submaps_strings.npy"), submap_strings_array)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Segmenting and Optimizing Implicit Map")
    parser.add_argument("configuration_path")
    parser.add_argument("--gpu_ids", nargs="*", required=False, default = ["0"], help="Which GPUs to use")

    args = parser.parse_args()

    gpu_id = str(args.gpu_ids[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    optimize_and_segment_implicit_map(args.configuration_path)