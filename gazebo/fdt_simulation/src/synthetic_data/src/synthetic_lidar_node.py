#!/usr/bin/env python
# coding: utf-8

CHUNK_SIZE=512  

import rospy
from sensor_msgs.msg import PointCloud2
import random
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from ctypes import * # convert float to uint32
import open3d as o3d
import tf2_ros
from geometry_msgs.msg import PoseStamped
import os
import sys
import torch
import pathlib
import pickle
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))

position = PROJECT_ROOT.find("/LonerSLAM")

if position != -1:
    PROJECT_ROOT = PROJECT_ROOT[:position + len("/LonerSLAM")]

sys.path.append(PROJECT_ROOT)

data_path = '/'.join(PROJECT_ROOT.split(os.sep)[:4]) + '/data'
output_path = '/'.join(PROJECT_ROOT.split(os.sep)[:4]) + '/outputs'
src_path = '/'.join(PROJECT_ROOT.split(os.sep)[:4]) + '/src'
analysis_path = '/'.join(PROJECT_ROOT.split(os.sep)[:4]) + '/analysis'
example_path = '/'.join(PROJECT_ROOT.split(os.sep)[:4]) + '/gazebo/example_implicit_map'

checkpoint = "final.tar"

exp_dir = example_path

sys.path.append(".")
sys.path.append(PROJECT_ROOT)
sys.path.append(src_path)
sys.path.append(analysis_path)
sys.path.append(analysis_path + "/fdt_common_utils")

from analysis.fdt_common_utils import build_lidar_scan
from analysis import fdt_render_dataset_frame

from src.common.pose_utils import WorldCube
from src.common.pose import Pose
from src.common.ray_utils import LidarRayDirections
from src.models.model_tcnn import Model, OccupancyGridModel
from src.models.ray_sampling import OccGridRaySampler


# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]
FIELDS_XYZRGB = FIELDS_XYZ + \
    [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]

# Bit operations
BIT_MOVE_16 = 2**16
BIT_MOVE_8 = 2**8
convert_rgbUint32_to_tuple = lambda rgb_uint32: (
    (rgb_uint32 & 0x00ff0000)>>16, (rgb_uint32 & 0x0000ff00)>>8, (rgb_uint32 & 0x000000ff)
)
convert_rgbFloat_to_tuple = lambda rgb_float: convert_rgbUint32_to_tuple(
    int(cast(pointer(c_float(rgb_float)), POINTER(c_uint32)).contents.value)
)

def convertCloudFromOpen3dToRos(open3d_cloud, frame_id="base"):
    """
    Converts a point cloud from Open3D format to ROS format.

    Args:
        open3d_cloud (open3d.geometry.PointCloud): The input point cloud in Open3D format.
        frame_id (str, optional): The frame ID for the ROS message. Defaults to "base".

    Returns:
        sensor_msgs.msg.PointCloud2: The converted point cloud in ROS format.
    """

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    points = np.asarray(open3d_cloud.points)

    if not open3d_cloud.colors: # XYZ only
        fields = FIELDS_XYZ
        cloud_data = points
    else: # XYZ + RGB
        fields = FIELDS_XYZRGB
        # Change rgb color from "three float" to "one 24-byte int"
        # 0x00FFFFFF is white, 0x00000000 is black.
        colors = np.floor(np.asarray(open3d_cloud.colors) * 255) # nx3 matrix
        colors = colors[:, 0] * BIT_MOVE_16 + colors[:, 1] * BIT_MOVE_8 + colors[:, 2]
        cloud_data = np.c_[points, colors]
    
    # Create the ROS point cloud message
    return pc2.create_cloud(header, fields, cloud_data)

def generate_random_pointcloud(num_points):
    """
    Generates a random point cloud with the specified number of points.
    For debuging purposes.
    Args:
        num_points (int): The number of points to generate.

    Returns:
        list: A list of points in the format [x, y, z, 0].
    """
    points = []
    for _ in range(num_points):
        x = random.uniform(-10, 10)
        y = random.uniform(-10, 10)
        z = random.uniform(-10, 10)
        points.append([x, y, z, 0])
    return points

if __name__ == '__main__':

    rospy.init_node('synthetic_lidar_node')
    rospy.logwarn('Launching Synthetic Lidar Node')

    rate = rospy.Rate(1.0)

    buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(buffer)

    checkpoint_path = pathlib.Path(f"{exp_dir}/checkpoints/{checkpoint}")

    with open(f"{exp_dir}/full_config.pkl", 'rb') as f:
        full_config = pickle.load(f)
    _DEVICE = torch.device(full_config.mapper.device)

    occ_model_config = full_config.mapper.optimizer.model_config.model.occ_model
    assert isinstance(occ_model_config, dict), f"OGM enabled but model.occ_model is empty"

    scale_factor = full_config.world_cube.scale_factor.to(_DEVICE)
    shift = full_config.world_cube.shift.to(_DEVICE)
    world_cube = WorldCube(scale_factor, shift).to(_DEVICE)
    occ_model = OccupancyGridModel(occ_model_config).to(_DEVICE)
    ray_sampler = OccGridRaySampler()

    model_config = full_config.mapper.optimizer.model_config.model
    model = Model(model_config).to(_DEVICE)

    print(f'Loading checkpoint from: {checkpoint_path}')
    ckpt = torch.load(str(checkpoint_path))
    model.load_state_dict(ckpt['network_state_dict'])

    occ_model.load_state_dict(ckpt['occ_model_state_dict'])
    occupancy_grid = occ_model()
    ray_sampler.update_occ_grid(occupancy_grid.detach())

    lidar_scan = build_lidar_scan.QT64()
    ray_directions = LidarRayDirections(lidar_scan, chunk_size=CHUNK_SIZE)

    ray_range = (2.5,45)

    model_data = (model, ray_sampler, world_cube, ray_range, _DEVICE, CHUNK_SIZE)

    while not rospy.is_shutdown():

        try:
            tf_msg = buffer.lookup_transform('base_link','hesai/pandar', rospy.Time(0), timeout=rospy.Duration(0.9))
            (translation, rotation) = (tf_msg.transform.translation,tf_msg.transform.rotation)

        except:
            rospy.logwarn('Failed to get transform from /base_link to /hesai/pandar')
            rate.sleep()
            continue

        rotation_euler = R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w]).as_euler('xyz', degrees=True)
        pose = np.hstack((np.array([translation.x, translation.y, translation.z]),rotation_euler))
        lidar_pose = Pose(pose_tensor=torch.from_numpy(pose).to(_DEVICE))
        rendered_lidar = fdt_render_dataset_frame.LiDAR(lidar_pose, ray_directions, model_data, var_threshold=0.1)
    
        ros_pcd_pub = rospy.Publisher("pcd", PointCloud2, queue_size=10)
        pose_pub = rospy.Publisher("world_frame", PoseStamped, queue_size=10)

        # pointcloud = generate_random_pointcloud(1000)

        open3d_cloud = o3d.geometry.PointCloud()
        open3d_cloud.points = o3d.utility.Vector3dVector(rendered_lidar)
        ros_pcd = convertCloudFromOpen3dToRos(open3d_cloud,"hesai/pandar")
        
        ros_pcd_pub.publish(ros_pcd)
        rate.sleep()