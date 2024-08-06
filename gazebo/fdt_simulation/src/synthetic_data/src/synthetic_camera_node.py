#!/usr/bin/env python
# coding: utf-8

CHUNK_SIZE=200
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import numpy as np
import tf2_ros
import os
import sys
import torch
import pathlib
import pickle
from scipy.spatial.transform import Rotation as R
import cv2
from cv_bridge import CvBridge


PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))

position = PROJECT_ROOT.find("/LonerSLAM")

# Slice the string up to the end of '/LonerSLAM'
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

print("------------------")
print(PROJECT_ROOT)

from analysis import fdt_render_dataset_frame

from src.common.pose_utils import WorldCube
from src.common.pose import Pose
from src.common.ray_utils import CameraRayDirections
from src.models.model_tcnn import Model, OccupancyGridModel
from src.models.ray_sampling import OccGridRaySampler

def convertCvImageToRos(cv_image, frame_id="camera"):
    """
    Converts an OpenCV image to a ROS image.

    Args:
        cv_image (np.ndarray): The OpenCV image to be converted.
        frame_id (str, optional): The frame ID of the ROS image. Defaults to "camera".

    Returns:
        sensor_msgs.msg.Image: The converted ROS image.
    """
    bridge = CvBridge()
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id
    ros_image = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
    ros_image.header = header
    return ros_image

if __name__ == '__main__':
    rospy.init_node('synthetic_camera_node')
    rospy.logwarn('Launching Synthetic Camera Node')

    rate = rospy.Rate(1.0)  # 1 Hz
    buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(buffer)

    checkpoint_path = pathlib.Path(f"{exp_dir}/checkpoints/{checkpoint}")

    with open(f"{exp_dir}/full_config.pkl", 'rb') as f:
        full_config = pickle.load(f)

    full_config["calibration"]["camera_intrinsic"]["k"] = torch.tensor(full_config["calibration"]["camera_intrinsic"]["k"])
    full_config["calibration"]["camera_intrinsic"]["new_k"] = full_config["calibration"]["camera_intrinsic"]["k"]
    full_config["calibration"]["camera_intrinsic"]["distortion"] = torch.tensor(full_config["calibration"]["camera_intrinsic"]["distortion"])
    full_config["calibration"]["lidar_to_camera"]["orientation"] = np.array(full_config["calibration"]["lidar_to_camera"]["orientation"]) # for weird compatability 
    full_config["calibration"]["lidar_to_camera"]["xyz"] = np.array(full_config["calibration"]["lidar_to_camera"]["xyz"])

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

    intrinsic = full_config.calibration.camera_intrinsic
    im_size = torch.Tensor([intrinsic.height, intrinsic.width])

    ray_directions = CameraRayDirections(full_config.calibration, chunk_size=CHUNK_SIZE, device=_DEVICE)
    ray_range = (2.5,45)

    model_data = (model, ray_sampler, world_cube, ray_range, _DEVICE, CHUNK_SIZE)

    while not rospy.is_shutdown():
        try:
            tf_msg = buffer.lookup_transform('base_link','camera/right', rospy.Time(0), timeout=rospy.Duration(0.9))
            (translation, rotation) = (tf_msg.transform.translation, tf_msg.transform.rotation)
        except:
            rospy.logwarn('Failed to get transform from /base_link to /camera/right')
            rate.sleep()
            continue

        rotation_euler = R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w]).as_euler('xyz', degrees=True)
        pose = np.hstack((np.array([translation.x, translation.y, translation.z]), rotation_euler))

        lidar_to_camera = Pose.from_settings(full_config.calibration.lidar_to_camera)
        lidar_pose = Pose(pose_tensor=torch.from_numpy(pose).to(_DEVICE))
        cam_pose = lidar_pose.to(_DEVICE) * lidar_to_camera.to(_DEVICE)

        # dummy image for debuging purposes
        # height, width = intrinsic.height, intrinsic.width
        # random_image_ = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

        synthetic_image,_,_,_,_,_ = fdt_render_dataset_frame.RGBD(cam_pose, im_size, ray_directions, model_data)
        synthetic_image_ = synthetic_image.squeeze().permute(1, 2, 0).detach().cpu().numpy()*255

        ros_image_pub = rospy.Publisher("camera/image", Image, queue_size=10)
        ros_image = convertCvImageToRos(synthetic_image_.astype(np.uint8) , "camera_link")
        
        ros_image_pub.publish(ros_image)
        rate.sleep()

