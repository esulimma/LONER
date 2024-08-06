import os
import rosbag
import numpy as np
from scipy.spatial.transform import Rotation as R
import pytorch3d.transforms
import torch
import csv

"""
topics:                   
/tf                                               : tf2_msgs/TFMessage               (2 connections)
/tf_static                                         : tf2_msgs/TFMessage                    
/zed2i/zed_node/rgb/camera_info                    : sensor_msgs/CameraInfo          
"""

def get_calibration(bag_file):
    static_tf(bag_file)
    camera_info(bag_file)

def camera_info(bag_file, topic='/zed2i/zed_node/rgb/camera_info', save_all=False):
    """
    Extracts camera calibration information from a ROS bag file and saves it as a CSV file.

    Args:
        bag_file (str): The path to the ROS bag file.
        topic (str, optional): The topic to extract camera info from. Defaults to '/zed2i/zed_node/rgb/camera_info'.
        save_all (bool, optional): Whether to save all camera info messages to a separate CSV file. Defaults to False.
    Returns:
        None
    """

    output_dir_path = os.path.dirname(bag_file)
    bag_file_name = os.path.basename(bag_file)

    output_path = output_dir_path + "/calibration_" + bag_file_name[:-4] + topic.replace('/', '_')
    output_csv = output_path + '.csv'

    fieldnames = ['width', 'height', 'distortion_model', 'D', 'K', 'R', 'P']

    with rosbag.Bag(bag_file, 'r') as bag:
        # Extract camera info from the specified topic
        for _, msg, _ in bag.read_messages(topics=[topic]):
            fieldvalues = [msg.width,
                           msg.height,
                           msg.distortion_model,
                           msg.D,
                           msg.K,
                           msg.R,
                           msg.P]
            break

    # Save camera info as a CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i, name in enumerate(fieldnames):
            writer.writerow([name + ':'])
            writer.writerow([fieldvalues[i]])

    # Save camera info as a numpy array
    np.save(output_path, [fieldnames, fieldvalues])

    # Print camera calibration information
    print("---------------------------------------------------------------------")
    print("calibration:")
    print("  camera_intrinsic:")
    print(f"    k: [[{msg.K[0]}, {msg.K[1]}, {msg.K[2]}],[{msg.K[3]}, {msg.K[4]}, {msg.K[5]}],[{msg.K[6]}, {msg.K[7]}, {msg.K[8]}]]")
    print(f"    distortion: [{msg.D[0]}, {msg.D[1]}, {msg.D[2]}, {msg.D[3]}, {msg.D[4]}]")
    print(f"    new_k: NULL")
    print(f"    width: {msg.width}")
    print(f"    height: {msg.height}")

    if save_all:
        # Save all camera info messages to a separate CSV file
        output_name = "/messages_" + bag_file_name[:-4] + topic.replace('/', '_') + '.csv'
        output_csv = output_dir_path + output_name
        with rosbag.Bag(bag_file, 'r') as bag:
            with open(output_csv, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'width', 'height', 'distortion_model', 'D', 'K', 'R', 'P'])  # CSV header

                for topic, msg, t in bag.read_messages(topics=['/zed2i/zed_node/rgb/camera_info']):
                    writer.writerow([msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9,
                                     msg.width,
                                     msg.height,
                                     msg.distortion_model,
                                     msg.D,
                                     msg.K,
                                     msg.R,
                                     msg.P])
                    
def static_tf(bag_file, tf_topic="/tf_static", print_tf=False):
    """
    Extracts static transformations from a ROS bag file.

    Args:
        bag_file (str): Path to the ROS bag file.
        tf_topic (str, optional): Topic name for the static transformations. Defaults to "/tf_static".
    Returns:
        None
    """

    # Open the bag file
    bag = rosbag.Bag(bag_file)

    # Loop through messages in the bag and extract static transformations
    for _, msg, _ in bag.read_messages(topics = [tf_topic]):
        for transform in msg.transforms:
            if transform.header.frame_id == "main_sensor_setup" and transform.child_frame_id == "PandarQT_base":
                t_main_sensor_setup_PandarQT_base = np.array([[transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]])
                r_main_sensor_setup_PandarQT_base = R.from_quat([[transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]]).as_matrix()
                R_main_sensor_setup_PandarQT_base = np.hstack((np.vstack((r_main_sensor_setup_PandarQT_base.squeeze(), np.zeros((1, 3)))),np.hstack((t_main_sensor_setup_PandarQT_base, np.ones((1, 1)))).T))

            if transform.header.frame_id == "PandarQT_base" and transform.child_frame_id == "PandarQT":
                t_PandarQT_base_PandarQT = np.array([[transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]])
                r_PandarQT_base_PandarQT = R.from_quat([[transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]]).as_matrix()
                R_PandarQT_base_PandarQT = np.hstack((np.vstack((r_PandarQT_base_PandarQT.squeeze(), np.zeros((1, 3)))),np.hstack((t_PandarQT_base_PandarQT, np.ones((1, 1)))).T))
                                                     
            if transform.header.frame_id == "main_sensor_setup" and transform.child_frame_id == "zed2i_base_link":
                t_main_sensor_setup_zed2i_base_link = np.array([[transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]])
                r_main_sensor_setup_zed2i_base_link = R.from_quat([[transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]]).as_matrix()
                R_main_sensor_setup_zed2i_base_link = np.hstack((np.vstack((r_main_sensor_setup_zed2i_base_link.squeeze(), np.zeros((1, 3)))),np.hstack((t_main_sensor_setup_zed2i_base_link, np.ones((1, 1)))).T))
                                                                
            if transform.header.frame_id == "zed2i_base_link" and transform.child_frame_id == "zed2i_camera_center":
                t_zed2i_base_link_zed2i_camera_center = np.array([[transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]])
                r_zed2i_base_link_zed2i_camera_center = R.from_quat([[transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]]).as_matrix()
                R_zed2i_base_link_zed2i_camera_center = np.hstack((np.vstack((r_zed2i_base_link_zed2i_camera_center.squeeze(), np.zeros((1, 3)))),np.hstack((t_zed2i_base_link_zed2i_camera_center, np.ones((1, 1)))).T))
                                                                  
            if transform.header.frame_id == "zed2i_camera_center" and transform.child_frame_id == "zed2i_right_camera_frame":
                t_zed2i_camera_center_zed2i_right_camera_frame = np.array([[transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]])
                r_zed2i_camera_center_zed2i_right_camera_frame = R.from_quat([[transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]]).as_matrix()
                R_zed2i_camera_center_zed2i_right_camera_frame = np.hstack((np.vstack((r_zed2i_camera_center_zed2i_right_camera_frame.squeeze(), np.zeros((1, 3)))),np.hstack((t_zed2i_camera_center_zed2i_right_camera_frame, np.ones((1, 1)))).T))
                                                                           
            if transform.header.frame_id == "zed2i_right_camera_frame" and transform.child_frame_id == "zed2i_right_camera_optical_frame":
                t_zed2i_right_camera_frame_zed2i_right_camera_optical_frame = np.array([[transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]])
                r_zed2i_right_camera_frame_zed2i_right_camera_optical_frame =R.from_quat([[transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]]).as_matrix()
                R_zed2i_right_camera_frame_zed2i_right_camera_optical_frame = np.hstack((np.vstack((r_zed2i_right_camera_frame_zed2i_right_camera_optical_frame.squeeze(), np.zeros((1, 3)))),np.hstack((t_zed2i_right_camera_frame_zed2i_right_camera_optical_frame, np.ones((1, 1)))).T))
                                                                                        
    r_PandarQT_main_sensor_setup = R.from_matrix(np.matmul(r_main_sensor_setup_PandarQT_base, r_PandarQT_base_PandarQT)).inv().as_matrix()

    r_PandarQT_base_zed2i_base_link = np.matmul(r_PandarQT_main_sensor_setup, r_main_sensor_setup_zed2i_base_link)
    r_PandarQT_base_zed2i_camera_center = np.matmul(r_PandarQT_base_zed2i_base_link, r_zed2i_base_link_zed2i_camera_center)
    r_PandarQT_base_zed2i_right_camera_frame = np.matmul(r_PandarQT_base_zed2i_camera_center, r_zed2i_camera_center_zed2i_right_camera_frame)
    r_PandarQT_base_zed2i_right_camera_optical_frame = np.matmul(r_PandarQT_base_zed2i_right_camera_frame, r_zed2i_right_camera_frame_zed2i_right_camera_optical_frame)

    t_PandarQT_main_sensor_setup = np.matmul(r_PandarQT_base_PandarQT, t_PandarQT_base_PandarQT.T) + t_main_sensor_setup_PandarQT_base
    
    t_PandarQT_base_zed2i_base_link = np.matmul(r_PandarQT_main_sensor_setup, r_main_sensor_setup_zed2i_base_link)
    t_PandarQT_base_zed2i_camera_center = np.matmul(r_PandarQT_base_zed2i_base_link, r_zed2i_base_link_zed2i_camera_center)
    t_PandarQT_base_zed2i_right_camera_frame = np.matmul(r_PandarQT_base_zed2i_camera_center, r_zed2i_camera_center_zed2i_right_camera_frame)
    t_PandarQT_base_zed2i_right_camera_optical_frame = np.matmul(r_PandarQT_base_zed2i_right_camera_frame, r_zed2i_right_camera_frame_zed2i_right_camera_optical_frame)

    t_PandarQT_base_zed2i_right_camera_optical_frame = - t_PandarQT_base_PandarQT -t_main_sensor_setup_PandarQT_base + t_main_sensor_setup_zed2i_base_link + t_zed2i_base_link_zed2i_camera_center + t_zed2i_camera_center_zed2i_right_camera_frame + t_zed2i_right_camera_frame_zed2i_right_camera_optical_frame
    
    R_PandarQT_main_sensor_setup = np.linalg.inv(np.matmul(R_main_sensor_setup_PandarQT_base, R_PandarQT_base_PandarQT))

    R_PandarQT_base_zed2i_base_link = np.matmul(R_PandarQT_main_sensor_setup, R_main_sensor_setup_zed2i_base_link)
    R_PandarQT_base_zed2i_camera_center = np.matmul(R_PandarQT_base_zed2i_base_link, R_zed2i_base_link_zed2i_camera_center)
    R_PandarQT_base_zed2i_right_camera_frame = np.matmul(R_PandarQT_base_zed2i_camera_center, R_zed2i_camera_center_zed2i_right_camera_frame)
    R_PandarQT_base_zed2i_right_camera_optical_frame = np.matmul(R_PandarQT_base_zed2i_right_camera_frame, R_zed2i_right_camera_frame_zed2i_right_camera_optical_frame)

    if print_tf:
        print("---------------------------------------------------------------------")
        print("main_sensor_setup_PandarQT_base:") 
        print(f"translation: {t_main_sensor_setup_PandarQT_base}")
        print(f"rotation: {R.from_matrix(r_main_sensor_setup_PandarQT_base).as_euler('xyz', degrees=True)}")
        print("PandarQT_base_PandarQT:")
        print(f"translation: {t_PandarQT_base_PandarQT}")
        print(f"rotation: {R.from_matrix(r_PandarQT_base_PandarQT).as_euler('xyz', degrees=True)}")
        print("main_sensor_setup_zed2i_base_link:")
        print(f"translation: {t_main_sensor_setup_zed2i_base_link}")
        print(f"rotation: {R.from_matrix(r_main_sensor_setup_zed2i_base_link).as_euler('xyz', degrees=True)}")
        print("zed2i_base_link_zed2i_camera_center:")
        print(f"translation: {t_zed2i_base_link_zed2i_camera_center}")
        print(f"rotation: {R.from_matrix(r_zed2i_base_link_zed2i_camera_center).as_euler('xyz', degrees=True)}")
        print("zed2i_camera_center_zed2i_right_camera_frame:")
        print(f"translation: {t_zed2i_camera_center_zed2i_right_camera_frame}")
        print(f"rotation: {R.from_matrix(r_zed2i_camera_center_zed2i_right_camera_frame).as_euler('xyz', degrees=True)}")
        print("zed2i_right_camera_frame_zed2i_right_camera_optical_frame:")
        print(f"translation: {t_zed2i_right_camera_frame_zed2i_right_camera_optical_frame}")
        print(f"rotation: {R.from_matrix(r_zed2i_right_camera_frame_zed2i_right_camera_optical_frame).as_euler('xyz', degrees=True)}")

        print("")
        print("PandarQT_main_sensor_setup:")
        print(f"translation: {t_PandarQT_main_sensor_setup}")
        print(f"rotation: {R.from_matrix(r_PandarQT_main_sensor_setup).as_euler('xyz', degrees=True)}")
        print(f"rotation matrix:")
        print(r_PandarQT_main_sensor_setup)    

        print("")
        print("PandarQT_main_sensor_setup from R:")
        print(f"translation: {R_PandarQT_main_sensor_setup[0:3,3]}")
        print(f"rotation: {R.from_matrix(R_PandarQT_main_sensor_setup[0:3,0:3]).as_euler('xyz', degrees=True)}")
        print(f"rotation matrix:")
        print(R_PandarQT_main_sensor_setup[0:3,0:3])    

        print("")
        print("PandarQT_base_zed2i_right_camera_optical_frame:")
        print(f"translation: {t_PandarQT_base_zed2i_right_camera_optical_frame}")
        print(f"rotation: {R.from_matrix(r_PandarQT_base_zed2i_right_camera_optical_frame).as_euler('XYZ', degrees=True)}") 
        print(f"rotation matrix:")
        print(R_PandarQT_base_zed2i_right_camera_optical_frame[0:3,0:3])

        print("")
        print("PandarQT_base_zed2i_right_camera_optical_frame from R:")
        print(f"translation: {R_PandarQT_base_zed2i_right_camera_optical_frame[0:3,3]}")
        print(f"rotation XYZ: {R.from_matrix(R_PandarQT_base_zed2i_right_camera_optical_frame[0:3,0:3]).as_euler('XYZ', degrees=True)}")
        print(f"rotation xyz: {R.from_matrix(R_PandarQT_base_zed2i_right_camera_optical_frame[0:3,0:3]).as_euler('xyz', degrees=True)}")
        print(f"rotation matrix:")
        print(R_PandarQT_base_zed2i_right_camera_optical_frame[0:3,0:3])

        print("")

        print("Pose Tensor:")
        print(f"    translation: {R_PandarQT_base_zed2i_right_camera_optical_frame[0:3,3]}")
        print(f"    orientation: {pytorch3d.transforms.quaternion_to_axis_angle(torch.from_numpy(R.from_matrix(R_PandarQT_base_zed2i_right_camera_optical_frame[0:3,0:3]).as_quat())).numpy()}")

        print("")

    print("---------------------------------------------------------------------")
    print("calibration:")
    print("  lidar_to_camera:")
    print(f"    xyz: {R_PandarQT_base_zed2i_right_camera_optical_frame[0:3,3]}")
    print(f"    orientation: {-R.from_matrix(R_PandarQT_base_zed2i_right_camera_optical_frame[0:3,0:3]).as_quat()}")
    # print(f"    orientation: {R.from_matrix(R_PandarQT_base_zed2i_right_camera_optical_frame[0:3,0:3]).as_quat()}")


# transforms: 
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "BASE_FORE"
#     child_frame_id: "BASE_inertia"
#     transform: 
#       translation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "main_sensor_setup"
#     child_frame_id: "GNSS"
#     transform: 
#       translation: 
#         x: 0.0
#         y: 0.0
#         z: 0.1965
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "BASE"
#     child_frame_id: "IMU_BASE_base_link"
#     transform: 
#       translation: 
#         x: 0.021600000000000064
#         y: 0.0
#         z: 0.1
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "IMU_BASE_base_link"
#     child_frame_id: "IMU_BASE_link"
#     transform: 
#       translation: 
#         x: -0.0107
#         y: -0.00505
#         z: -0.00705
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "BASE"
#     child_frame_id: "BASE_FORE"
#     transform: 
#       translation: 
#         x: 1.67
#         y: 0.0
#         z: 0.0
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "LF_WHEEL"
#     child_frame_id: "LF_WHEEL_CONTACT"
#     transform: 
#       translation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "LH_WHEEL"
#     child_frame_id: "LH_WHEEL_CONTACT"
#     transform: 
#       translation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "main_sensor_setup"
#     child_frame_id: "PandarQT_base"
#     transform: 
#       translation: 
#         x: 0.0
#         y: 0.0
#         z: 0.114
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.706825181105366
#         w: 0.7073882691671998
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "RF_WHEEL"
#     child_frame_id: "RF_WHEEL_CONTACT"
#     transform: 
#       translation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "RH_WHEEL"
#     child_frame_id: "RH_WHEEL_CONTACT"
#     transform: 
#       translation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "TOOL_TILT"
#     child_frame_id: "TOOL_CONTACT"
#     transform: 
#       translation: 
#         x: 0.5
#         y: 0.0
#         z: 0.0
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "imu_perception_kit_base"
#     child_frame_id: "imu_perception_kit_link"
#     transform: 
#       translation: 
#         x: 0.00071
#         y: -0.0055
#         z: 0.0
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "main_sensor_setup"
#     child_frame_id: "imu_perception_kit_base"
#     transform: 
#       translation: 
#         x: 0.0
#         y: 0.082
#         z: 0.114
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "PandarQT_base"
#     child_frame_id: "PandarQT"
#     transform: 
#       translation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0504
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "sensor_mount"
#     child_frame_id: "main_sensor_setup"
#     transform: 
#       translation: 
#         x: 0.0
#         y: -0.573
#         z: 0.215
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "BASE_REAR"
#     child_frame_id: "pc_box"
#     transform: 
#       translation: 
#         x: 0.0
#         y: 0.0
#         z: 0.96
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "BOOM_BASE"
#     child_frame_id: "sensor_mount"
#     transform: 
#       translation: 
#         x: -0.166
#         y: -0.113
#         z: 0.479
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "zed2i_camera_center"
#     child_frame_id: "zed2i_baro_link"
#     transform: 
#       translation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "main_sensor_setup"
#     child_frame_id: "zed2i_base_link"
#     transform: 
#       translation: 
#         x: 0.19829
#         y: 0.0
#         z: 0.02506
#       rotation: 
#         x: 0.0
#         y: 0.13528341350446557
#         z: 0.0
#         w: 0.9908069428655514
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "zed2i_base_link"
#     child_frame_id: "zed2i_camera_center"
#     transform: 
#       translation: 
#         x: 0.0
#         y: 0.0
#         z: 0.015
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "zed2i_camera_center"
#     child_frame_id: "zed2i_left_camera_frame"
#     transform: 
#       translation: 
#         x: 0.0
#         y: 0.06
#         z: 0.0
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "zed2i_left_camera_frame"
#     child_frame_id: "zed2i_left_camera_optical_frame"
#     transform: 
#       translation: 
#         x: -0.01
#         y: 0.0
#         z: 0.0
#       rotation: 
#         x: 0.5
#         y: -0.4999999999999999
#         z: 0.5
#         w: -0.5000000000000001
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "zed2i_camera_center"
#     child_frame_id: "zed2i_mag_link"
#     transform: 
#       translation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "zed2i_camera_center"
#     child_frame_id: "zed2i_right_camera_frame"
#     transform: 
#       translation: 
#         x: 0.0
#         y: -0.06
#         z: 0.0
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "zed2i_right_camera_frame"
#     child_frame_id: "zed2i_right_camera_optical_frame"
#     transform: 
#       translation: 
#         x: -0.01
#         y: 0.0
#         z: 0.0
#       rotation: 
#         x: 0.5
#         y: -0.4999999999999999
#         z: 0.5
#         w: -0.5000000000000001
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "zed2i_left_camera_frame"
#     child_frame_id: "zed2i_temp_left_link"
#     transform: 
#       translation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0
#   - 
#     header: 
#       seq: 0
#       stamp: 
#         secs: 1683180302
#         nsecs: 589937425
#       frame_id: "zed2i_right_camera_frame"
#     child_frame_id: "zed2i_temp_right_link"
#     transform: 
#       translation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#       rotation: 
#         x: 0.0
#         y: 0.0
#         z: 0.0
#         w: 1.0