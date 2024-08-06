import rosbag
import csv
import os

"""
topics:       
/harveri_base_imu                                  : sensor_msgs/Imu                 
/imu_perception_kit                                : sensor_msgs/Imu                 
/tf                                               : tf2_msgs/TFMessage               (2 connections)
/tf_static                                         : tf2_msgs/TFMessage              
/mapping_node/scan2map_odometry    281 msgs    : nav_msgs/Odometry
/mapping_node/scan2scan_odometry   517 msgs    : nav_msgs/Odometry
"""

def extract_odom_o3d(bag_file, output_file, odom_topic="/mapping_node/scan2scan_odometry"):
    """
    Extracts odometry data from a ROS bag file and writes it to a CSV file.

    Parameters:
    - bag_file (str): Path to the ROS bag file.
    - output_file (str): Path to the output CSV file.
    - odom_topic (str): ROS topic containing the odometry messages. Default is "/mapping_node/scan2scan_odometry".
    Returns:
        None
    """
    with open(output_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['timestamp', 'linear_translation_x', 'linear_translation_y', 'linear_translation_z', 'rotation_x', 'rotation_y', 'rotation_z'])
        # Read messages from the bag file
        bag = rosbag.Bag(bag_file)
        first_twist = None
        first_timestamp = None
        for _, msg, _ in bag.read_messages(topics=[odom_topic]):
            twist = msg.twist.twist
            if first_twist is None:
                first_twist = twist
                first_timestamp = msg.header.stamp.to_sec()       
            timestamp = round(msg.header.stamp.to_sec() - first_timestamp, 10)
            translation = twist.linear
            rotation = twist.angular
            # Write to CSV
            values = [timestamp, translation.x - first_twist.linear.x, translation.y - first_twist.linear.y, translation.z - first_twist.linear.z, rotation.x, rotation.y, rotation.z]
            writer.writerow(values)

def extract_gt_o3d_rpy(bag_file, output_file, tf_topic="/mapping_node/scan2map_transform"):
    """
    Extracts ground truth trajectories from a ROS bag file and writes them to a CSV file.

    Parameters:
    - bag_file (str): Path to the ROS bag file.
    - output_file (str): Path to the output CSV file.
    - tf_topic (str): ROS topic containing the transform messages. Default is "/mapping_node/scan2map_transform".
    Returns:
        None
    """
    with open(output_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['timestamp', 'linear_translation_x', 'linear_translation_y', 'linear_translation_z', 'rotation_x', 'rotation_y', 'rotation_z'])     
        # Read messages from the bag file
        bag = rosbag.Bag(bag_file)
        first_transform = None
        first_timestamp = None
        for _, msg, _ in bag.read_messages(topics=[tf_topic]):
            transform = msg.transform
            if first_transform is None:
                first_transform = transform
                first_timestamp = msg.header.stamp.to_sec()
            timestamp = round(msg.header.stamp.to_sec() - first_timestamp, 10)
            translation = transform.translation
            rotation = transform.rotation
            # Write to CSV
            values = [timestamp, translation.x - first_transform.translation.x, translation.y - first_transform.translation.y, translation.z - first_transform.translation.z, rotation.x, rotation.y, rotation.z]
            writer.writerow(values)

def extract_gt_o3d_tum(bag_file, output_file = None, tf_topic="/mapping_node/scan2map_transform", remove_offset=False):
    """
    Extracts ground truth trajectories from a rosbag file and saves them in a CSV file.

    Parameters:
    - bag_file (str): Path to the rosbag file.
    - output_file (str): Path to the output CSV file.
    - tf_topic (str, optional): ROS topic containing the transform messages. Default is "/mapping_node/scan2map_transform".
    - remove_offset (bool, optional): Whether to remove the offset from the first transform. Default is False.
    Returns:
        None
    """
    if output_file is None:
        output_file = os.path.dirname(bag_file) +  '/ground_truth_traj.csv'
    with open(output_file, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=' ')     
        # Read messages from the bag file
        bag = rosbag.Bag(bag_file)
        if remove_offset:
            first_transform = None
            first_timestamp = None
        else:
            first_timestamp = 0
        # writer.writerow(['# timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw'])
        for _, msg, _ in bag.read_messages(topics=[tf_topic]):
            transform = msg.transform
            if first_timestamp is None:
                first_transform = transform
                first_timestamp = msg.header.stamp.to_sec()
            t = msg.header.stamp
            timestamp = round(t.to_sec() - first_timestamp, 10)
            translation = transform.translation
            rotation = transform.rotation
            # Write to CSV
            if remove_offset:
                values = [timestamp, translation.x - first_transform.translation.x, translation.y - first_transform.translation.y, translation.z - first_transform.translation.z, rotation.x, rotation.y, rotation.z, rotation.w]
            else:
                values = [timestamp, translation.x, translation.y, translation.z, rotation.x, rotation.y, rotation.z, rotation.w]
            writer.writerow(values)

def extract_gt(bag_file, output_file, tf_topic="/tf"):
    """
    Extracts ground truth trajectory from a ROS bag file and saves it to a CSV file.

    Parameters:
    - bag_file (str): Path to the ROS bag file.
    - output_file (str): Path to the output CSV file.
    - tf_topic (str): Topic name for the tf messages in the ROS bag file. Default is "/tf".
    Returns:
        None
    """
    # Open the output CSV file
    with open(output_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['timestamp', 'linear_translation_x', 'linear_translation_y', 'linear_translation_z', 'rotation_x', 'rotation_y', 'rotation_z'])
        # Read the ROS bag file
        bag = rosbag.Bag(bag_file)
        first_transform = None
        first_timestamp = None
        # Iterate over the tf messages in the bag file
        for _, msg, _ in bag.read_messages(topics=[tf_topic]):
            timestamp = msg.header.stamp
            for transform in msg.transforms:
                if transform.header.frame_id == "odom":
                    if first_transform is None:
                        first_transform = transform
                        first_timestamp = msg.header.stamp
                    timestamp = round(msg.header.stamp.to_sec() - first_timestamp, 10)
                    translation = transform.transform.translation
                    rotation = transform.transform.rotation

                    # Write to CSV
                    values = [timestamp, translation.x - first_transform.transform.translation.x, translation.y - first_transform.transform.translation.y, translation.z - first_transform.transform.translation.z, rotation.x, rotation.y, rotation.z]
                    writer.writerow(values)

def extract_imu(bag_file, output_file, imu_topic = "/imu_perception_kit"):
    """
    Extracts IMU data from a rosbag file and saves it in a CSV file.

    Parameters:
    - bag_file (str): Path to the rosbag file.
    - output_file (str): Path to the output CSV file.
    - imu_topic (str, optional): ROS topic containing the IMU messages. Default is "/imu_perception_kit".
    Returns:
        None
    """
    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['timestamp', 'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z', 'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z'])
        # Read messages from the bag file
        bag = rosbag.Bag(bag_file)
        for _, msg, _ in bag.read_messages(topics=[imu_topic]):
            timestamp = ".".join([str(msg.header.stamp.secs), str(msg.header.stamp.nsecs)])
            linear_acceleration = msg.linear_acceleration
            angular_velocity = msg.angular_velocity
            # Write to CSV
            values = [timestamp, linear_acceleration.x, linear_acceleration.y, linear_acceleration.z, angular_velocity.x, angular_velocity.y, angular_velocity.z]
            writer.writerow(values)

def extract_haveri(bag_file, output_file, haveri_topic = "/harveri_measurements"):
    """
    Extracts haveri measurements from a rosbag file and saves them in a CSV file.

    Parameters:
    - bag_file (str): Path to the rosbag file.
    - output_file (str): Path to the output CSV file.
    - haveri_topic (str, optional): ROS topic containing the haveri measurements. Default is "/harveri_measurements".
    Returns:
        None
    """
    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['timestamp', 'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z', 'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z'])
        # Read messages from the bag file
        bag = rosbag.Bag(bag_file)
        for _, msg, _ in bag.read_messages(topics=[haveri_topic]):
            t = msg.header.stamp
            timestamp = ".".join([str(t.secs), str(t.nsecs)])
            linear_acceleration = msg.linear_acceleration
            angular_velocity = msg.angular_velocity
            # Write to CSV
            values = [timestamp, linear_acceleration.x, linear_acceleration.y, linear_acceleration.z, angular_velocity.x, angular_velocity.y, angular_velocity.z]
            writer.writerow(values)


def extract_empty_gt(bag_file, output_file, topic="/hesai/pandar"):
    """
    Extracts empty ground truth data from a ROS bag file and saves it to a text file.
    Used for running the realtime pipeline without ground truth data.

    Args:
        bag_file (str): Path to the ROS bag file.
        output_file (str): Path to the output text file.
    Returns:
        None
    """
    with open(output_file, 'w') as txt_file:
        timestamp_prev = ''.join("0.0 0 0 0 0 0 0 1")
        # Read messages from the bag file
        bag = rosbag.Bag(bag_file)
        for _, msg, _ in bag.read_messages(topics=[topic]):
            t = msg.header.stamp
            timestamp = ".".join([str(t.secs), str(t.nsecs)]) 
            if timestamp != timestamp_prev:
                # Write to CSV
                values = ''.join("0 0 0 0 0 0 1")
                txt_file.write(f"{timestamp} {(values)}\n")
            timestamp_prev = timestamp