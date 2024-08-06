import os
import sys

import adjust_bags
import extract_images
from image_segmentation import vit_h
from image_detect_keypoints import lucas_canade
from visualization_utils import split_image_gif
from visualization_utils import render_gif
from extract_calibration import get_calibration
from extract_trajectories import extract_gt_o3d_tum

############################################################

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)

data_path = os.path.dirname(PROJECT_ROOT) + '/data'

output_path = os.path.dirname(PROJECT_ROOT) + '/LonerSLAM/outputs'

############################################################

"""
topics:       
/harveri_base_imu                                  : sensor_msgs/Imu                 
/harveri_measurements                              : harveri_msgs/HarveriMeasurements
/hesai/pandar                                       : sensor_msgs/PointCloud2         
/imu_perception_kit                                : sensor_msgs/Imu                 
/tf                                               : tf2_msgs/TFMessage               (2 connections)
/tf_static                                         : tf2_msgs/TFMessage              
/zed2i/zed_node/confidence/confidence_map           : sensor_msgs/Image               
/zed2i/zed_node/depth/camera_info                   : sensor_msgs/CameraInfo          
/zed2i/zed_node/depth/depth_registered              : sensor_msgs/Image               
/zed2i/zed_node/point_cloud/cloud_registered        : sensor_msgs/PointCloud2         
/zed2i/zed_node/rgb/camera_info                    : sensor_msgs/CameraInfo          
/zed2i/zed_node/rgb/image_rect_color/compressed     : sensor_msgs/CompressedIma
"""
"""
/mapping_node/scan2map_odometry    281 msgs    : nav_msgs/Odometry
/mapping_node/scan2scan_odometry   517 msgs    : nav_msgs/Odometry
"""

############################################################

bag_file = data_path + "/haveri_hpk/02_02_04/harveri_hpk_02_02_04.bag"
GT_bag_file = data_path + "/haveri_hpk/02_02_04/harveri_hpk_02_02_04_GT.bag"

adjusted_input_bag_path = bag_file.replace('.bag', '_adjusted.bag')

images_folder = os.path.dirname(bag_file) + '/images/'
images_tracked_folder = os.path.dirname(bag_file) + '/images_tracked/'
images_segmented_folder = os.path.dirname(bag_file) + '/images_segmented/'
image_keypoints_folder = os.path.dirname(bag_file) + '/image_keypoints/'
image_mask_folder = os.path.dirname(bag_file) + '/image_mask/'

###############Adjust and Extract all####################
adjust_bags.adjust_and_extract_all(bag_file,GT_bag_file)

###############Adjust Lidar Scans and genereate Mask####################
adjust_bags.adjust_bag(bag_file)

###############Adjust Lidar Scans and genereate Mask####################
adjust_bags.read_and_adjust_scans(bag_file)

###############Extract Images####################
extract_images.save_compressed_images(bag_file)

###############Detect Keypoints####################

lucas_canade(images_folder, save_images=True)

###############Segment Images####################
vit_h(images_folder, save_images=True)

###############Get Calibration####################
get_calibration(bag_file)

###############Extract GT csv####################
extract_gt_o3d_tum(GT_bag_file)

###############Visualization####################
render_gif(images_folder)

images_folder = data_path + "/haveri_hpk/02_02_04/tryout/"
image1 = data_path + "/haveri_hpk/02_02_04/tryout/image1.png"
image2 = data_path + "/haveri_hpk/02_02_04/tryout/image2.png"
# images_folder = data_path + "/haveri_hpk/02_02_04/images/"

split_image_gif(image1,image2, images_folder)
""""""""""""""""""""""""""""""

images_folder = data_path + "/haveri_hpk/02_02_04/images/"
render_gif(images_folder)
""""""""""""""""""""""""""""""
# Bag names

bag_path = "/haveri_hpk/02_02_04/" # Relative to LonerSlamData Folder 

###############Append Rosbags####################

input_bag_names = ['harveri_hpk_2023-08-22-19-43-22_0.bag',
                   'harveri_hpk_2023-08-22-19-44-45_2.bag', 
                   'harveri_hpk_2023-08-22-19-45-23_3.bag', 
                   'harveri_hpk_2023-08-22-19-46-01_4.bag',
                   'harveri_hpk_2023-08-22-19-46-38_5.bag', 
                   'harveri_hpk_2023-08-22-19-45-23_3.bag', 
                   'harveri_hpk_2023-08-22-19-46-01_4.bag', 
                   'harveri_hpk_2023-08-22-19-46-38_5.bag', 
                   'harveri_hpk_2023-08-22-19-49-10_9.bag'
                   'harveri_hpk_2023-08-22-19-51-06_12.bag', 
                   'harveri_hpk_2023-08-22-19-53-00_15.bag']

input_bag_paths = [data_path + bag_path + name for name in input_bag_names]

out_bag_name = "harveri_hpk_2_2_4.bag"

input_bag_paths = [data_path + bag_path + name for name in input_bag_names]
output_bag_path = data_path + bag_path + out_bag_name

adjust_bags.append_bags(output_bag_path, input_bag_paths)

##### Example usage: Loop ####################

for input_bag_path in input_bag_paths:
    extract_images.read_and_adjust_scans(input_bag_path)

"""
topics:       
/harveri_base_imu                                  : sensor_msgs/Imu         
    header: 
    seq: 49689
    stamp: 
        secs: 1683206763
        nsecs: 281838720
    frame_id: ''
    orientation: 
    x: 0.79937744140625
    y: 0.600799560546875
    z: 0.003448486328124963
    w: 0.003082275390625049
    orientation_covariance: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    angular_velocity: 
    x: -0.027
    y: -0.002
    z: 0.006
    angular_velocity_covariance: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    linear_acceleration: 
    x: -0.08
    y: 0.35000000000000003
    z: -9.91
    linear_acceleration_covariance: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]        
/harveri_measurements                              : harveri_msgs/HarveriMeasurements
    header: 
    seq: 49689
    stamp: 
        secs: 1683206763
        nsecs: 278982315
    frame_id: ''
    actuator_states: 
    - 
        position: -0.048869219055841226
        velocity: 0.0
        effort: 0.0
        pressure_a: 0.0
        pressure_b: 0.0
        status: 3
    - 
        position: -0.2949078064710145
        velocity: -0.0003834951969714103
        effort: 29195.865109367267
        pressure_a: 13730000.0
        pressure_b: 6480000.0
        status: 0
    - 
        position: 0.0
        velocity: 0.0
        effort: 0.0
        pressure_a: 0.0
        pressure_b: 0.0
        status: 0
    - 
        position: -0.4068884039866663
        velocity: -0.0003834951969714103
        effort: 14827.272745386505
        pressure_a: 5100000.0
        pressure_b: 510000.0
        status: 0
    - 
        position: 0.0
        velocity: 0.0
        effort: 0.0
        pressure_a: 0.0
        pressure_b: 0.0
        status: 0
    - 
        position: -0.29567479686495735
        velocity: 0.0
        effort: 4711.493626478416
        pressure_a: 2030000.0
        pressure_b: 770000.0
        status: 0
    - 
        position: 0.0
        velocity: 0.0
        effort: 0.0
        pressure_a: 0.0
        pressure_b: 0.0
        status: 0
    - 
        position: -0.20056798801604758
        velocity: 0.0
        effort: 6680.447552207647
        pressure_a: 8440000.0
        pressure_b: 9350000.0
        status: 0
    - 
        position: 0.0
        velocity: 0.0
        effort: 0.0
        pressure_a: 0.0
        pressure_b: 0.0
        status: 0
    - 
        position: 0.3497476279735565
        velocity: 0.0
        effort: 0.0
        pressure_a: 0.0
        pressure_b: 0.0
        status: 0
    - 
        position: -0.704132867998116
        velocity: 0.0
        effort: 0.0
        pressure_a: 0.0
        pressure_b: 0.0
        status: 0
    - 
        position: -5.342968534868126
        velocity: 0.0
        effort: 0.0
        pressure_a: 0.0
        pressure_b: 0.0
        status: 0
    - 
        position: -0.0
        velocity: 0.0
        effort: 0.0
        pressure_a: 0.0
        pressure_b: 0.0
        status: 0
    - 
        position: 1.3542568290894255
        velocity: 0.0
        effort: 0.0
        pressure_a: 0.0
        pressure_b: 0.0
        status: 0
    - 
        position: 0.08209566902721863
        velocity: 0.0
        effort: 0.0
        pressure_a: 0.0
        pressure_b: 0.0
        status: 0
    - 
        position: 0.0
        velocity: 0.0
        effort: 0.0
        pressure_a: 0.0
        pressure_b: 0.0
        status: 0
    - 
        position: 0.0
        velocity: 0.0
        effort: 0.0
        pressure_a: 0.0
        pressure_b: 0.0
        status: 0
/hesai/pandar                                       : sensor_msgs/PointCloud2         
/imu_perception_kit                                : sensor_msgs/Imu                
    header: 
    seq: 102309
    stamp: 
        secs: 1683206763
        nsecs: 293835787
    frame_id: "imu_perception_kit_link"
    orientation: 
    x: 0.0
    y: 0.0
    z: 0.0
    w: 0.0
    orientation_covariance: [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    angular_velocity: 
    x: -0.02600100822746754
    y: 0.0052892426028847694
    z: -0.00750727253034711
    angular_velocity_covariance: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    linear_acceleration: 
    x: 0.14226919412612915
    y: -0.47228139638900757
    z: 9.701679229736328
    linear_acceleration_covariance: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   
/tf                                               : tf2_msgs/TFMessage               (2 connections)
/tf_static                                         : tf2_msgs/TFMessage   
    transforms: 
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "BASE_FORE"
        child_frame_id: "BASE_inertia"
        transform: 
        translation: 
            x: 0.0
            y: 0.0
            z: 0.0
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "main_sensor_setup"
        child_frame_id: "GNSS"
        transform: 
        translation: 
            x: 0.0
            y: 0.0
            z: 0.1965
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "BASE"
        child_frame_id: "IMU_BASE_base_link"
        transform: 
        translation: 
            x: 0.021600000000000064
            y: 0.0
            z: 0.1
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "IMU_BASE_base_link"
        child_frame_id: "IMU_BASE_link"
        transform: 
        translation: 
            x: -0.0107
            y: -0.00505
            z: -0.00705
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "BASE"
        child_frame_id: "BASE_FORE"
        transform: 
        translation: 
            x: 1.67
            y: 0.0
            z: 0.0
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "LF_WHEEL"
        child_frame_id: "LF_WHEEL_CONTACT"
        transform: 
        translation: 
            x: 0.0
            y: 0.0
            z: 0.0
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "LH_WHEEL"
        child_frame_id: "LH_WHEEL_CONTACT"
        transform: 
        translation: 
            x: 0.0
            y: 0.0
            z: 0.0
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "main_sensor_setup"
        child_frame_id: "PandarQT_base"
        transform: 
        translation: 
            x: 0.0
            y: 0.0
            z: 0.114
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.706825181105366
            w: 0.7073882691671998
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "RF_WHEEL"
        child_frame_id: "RF_WHEEL_CONTACT"
        transform: 
        translation: 
            x: 0.0
            y: 0.0
            z: 0.0
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "RH_WHEEL"
        child_frame_id: "RH_WHEEL_CONTACT"
        transform: 
        translation: 
            x: 0.0
            y: 0.0
            z: 0.0
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "TOOL_TILT"
        child_frame_id: "TOOL_CONTACT"
        transform: 
        translation: 
            x: 0.5
            y: 0.0
            z: 0.0
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "imu_perception_kit_base"
        child_frame_id: "imu_perception_kit_link"
        transform: 
        translation: 
            x: 0.00071
            y: -0.0055
            z: 0.0
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "main_sensor_setup"
        child_frame_id: "imu_perception_kit_base"
        transform: 
        translation: 
            x: 0.0
            y: 0.082
            z: 0.114
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "PandarQT_base"
        child_frame_id: "PandarQT"
        transform: 
        translation: 
            x: 0.0
            y: 0.0
            z: 0.0504
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "sensor_mount"
        child_frame_id: "main_sensor_setup"
        transform: 
        translation: 
            x: 0.0
            y: -0.573
            z: 0.215
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "BASE_REAR"
        child_frame_id: "pc_box"
        transform: 
        translation: 
            x: 0.0
            y: 0.0
            z: 0.96
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "BOOM_BASE"
        child_frame_id: "sensor_mount"
        transform: 
        translation: 
            x: -0.166
            y: -0.113
            z: 0.479
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "zed2i_camera_center"
        child_frame_id: "zed2i_baro_link"
        transform: 
        translation: 
            x: 0.0
            y: 0.0
            z: 0.0
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "main_sensor_setup"
        child_frame_id: "zed2i_base_link"
        transform: 
        translation: 
            x: 0.19829
            y: 0.0
            z: 0.02506
        rotation: 
            x: 0.0
            y: 0.13528341350446557
            z: 0.0
            w: 0.9908069428655514
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "zed2i_base_link"
        child_frame_id: "zed2i_camera_center"
        transform: 
        translation: 
            x: 0.0
            y: 0.0
            z: 0.015
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "zed2i_camera_center"
        child_frame_id: "zed2i_left_camera_frame"
        transform: 
        translation: 
            x: 0.0
            y: 0.06
            z: 0.0
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "zed2i_left_camera_frame"
        child_frame_id: "zed2i_left_camera_optical_frame"
        transform: 
        translation: 
            x: -0.01
            y: 0.0
            z: 0.0
        rotation: 
            x: 0.5
            y: -0.4999999999999999
            z: 0.5
            w: -0.5000000000000001
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "zed2i_camera_center"
        child_frame_id: "zed2i_mag_link"
        transform: 
        translation: 
            x: 0.0
            y: 0.0
            z: 0.0
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "zed2i_camera_center"
        child_frame_id: "zed2i_right_camera_frame"
        transform: 
        translation: 
            x: 0.0
            y: -0.06
            z: 0.0
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "zed2i_right_camera_frame"
        child_frame_id: "zed2i_right_camera_optical_frame"
        transform: 
        translation: 
            x: -0.01
            y: 0.0
            z: 0.0
        rotation: 
            x: 0.5
            y: -0.4999999999999999
            z: 0.5
            w: -0.5000000000000001
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "zed2i_left_camera_frame"
        child_frame_id: "zed2i_temp_left_link"
        transform: 
        translation: 
            x: 0.0
            y: 0.0
            z: 0.0
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0
    - 
        header: 
        seq: 0
        stamp: 
            secs: 1683206763
            nsecs: 267175003
        frame_id: "zed2i_right_camera_frame"
        child_frame_id: "zed2i_temp_right_link"
        transform: 
        translation: 
            x: 0.0
            y: 0.0
            z: 0.0
        rotation: 
            x: 0.0
            y: 0.0
            z: 0.0
            w: 1.0           
/zed2i/zed_node/confidence/confidence_map           : sensor_msgs/Image               
/zed2i/zed_node/depth/camera_info                   : sensor_msgs/CameraInfo          
/zed2i/zed_node/depth/depth_registered              : sensor_msgs/Image               
/zed2i/zed_node/point_cloud/cloud_registered        : sensor_msgs/PointCloud2         
/zed2i/zed_node/rgb/camera_info                    : sensor_msgs/CameraInfo          
/zed2i/zed_node/rgb/image_rect_color/compressed     : sensor_msgs/CompressedIma
"""
