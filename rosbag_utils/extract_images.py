import os
import numpy as np
import rosbag
import cv2
from cv_bridge import CvBridge

"""
topics:                 
/zed2i/zed_node/confidence/confidence_map           : sensor_msgs/Image               
/zed2i/zed_node/depth/depth_registered              : sensor_msgs/Image                        
/zed2i/zed_node/rgb/image_rect_color/compressed     : sensor_msgs/CompressedImage
"""

def save_compressed_images(bag_file_path, topic = '/zed2i/zed_node/rgb/image_rect_color/compressed',
                           output_dir = None):
    """
    Save images from a ROS bag file to a folder.

    Args:
        bag_file_path (str): The path to the ROS bag file.
        topic (str): The topic name to extract  compressed images from.
        output_dir (str, optional): The directory to save the images. If not provided, 
                                   images will be saved in a folder named 'images' 
                                   in the same directory as the bag file.

    Returns:
        None
    """
    if output_dir is None:
        output_dir = os.path.dirname(bag_file_path)+ '/images/'
    if not(os.path.isdir(output_dir)):
        os.mkdir(output_dir) 
    print("---------------------------------------------------------------------")
    print(f"Extracting compressed images from {bag_file_path}")
    print(f"With ROS image topic: {topic}")
    counter = 0
    with rosbag.Bag(bag_file_path, 'r') as bag:
        for _, msg, _ in bag.read_messages(topics=[topic]):
            t = msg.header.stamp
            # Convert ROS image message to OpenCV image
            np_arr = np.frombuffer(msg.data, np.uint8)
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            n_str = str(t.nsecs)
            if len(n_str) < 9:
                n_str = n_str.zfill(9)
            name = f"/image_{t.secs}_" + n_str + ".jpg"
            cv2.imwrite(f"{output_dir}"+ name, image_np)
            counter += 1
    print(f"Extracted {counter} images to {output_dir}")

def save_images(bag_file_path, topic = '/zed2i/zed_node/rgb/image_rect_color', 
                output_dir = None):
    """
    Save images from a ROS bag file to a folder.

    Args:
        bag_file_path (str): The path to the ROS bag file.
        topic (str): The topic name to extract images from.
        output_dir (str, optional): The directory to save the images. If not provided, 
                                   images will be saved in a folder named 'images' 
                                   in the same directory as the bag file.

    Returns:
        None
    """
    if output_dir is None:
        output_dir = os.path.dirname(bag_file_path)+ '/images/'
    if not(os.path.isdir(output_dir)):
        os.mkdir(output_dir) 
    print("---------------------------------------------------------------------")
    print(f"Extracting images from {bag_file_path}")
    print(f"With ROS image topic: {topic}")
    counter = 0
    with rosbag.Bag(bag_file_path, 'r') as bag:
        bridge = CvBridge()
        for _, msg, _ in bag.read_messages(topics=[topic]):
            t = msg.header.stamp
            # Convert ROS image message to OpenCV image
            image = bridge.imgmsg_to_cv2(msg, desired_encoding= msg.encoding)
            n_str = str(t.nsecs)
            if len(n_str) < 9:
                n_str = n_str.zfill(9)
            name = f"/image_{t.secs}_" + n_str + ".jpg"
            cv2.imwrite(f"{output_dir}"+ name, image)
            counter += 1
    print(f"Extracted {counter} images to {output_dir}")

def save_depth_images_8UC1(bag_file_path, topic):
    """
    Save depth images from a ROS bag file to folder.

    Args:
        bag_file_path (str): The path to the ROS bag file.
        topic (str): The topic name to extract depth images from.

    Returns:
        None
    """
    # Create output directory
    output_dir = bag_file_path.replace('.bag', '_depth_images_8UC1/')
    if not(os.path.isdir(output_dir)):
        os.mkdir(output_dir) 

    # Read messages from the bag file
    with rosbag.Bag(bag_file_path, 'r') as bag:
        bridge = CvBridge()
        for _, msg, _ in bag.read_messages(topics=[topic]):
            t = msg.header.stamp
            # Convert ROS image message to OpenCV image with 8UC1 encoding
            image = bridge.imgmsg_to_cv2(msg, desired_encoding='8UC1')
            # Save the depth image
            cv2.imwrite(f"{output_dir}/depth_{t.secs}_{t.nsecs}.png", image)

def save_depth_images_32FC1(bag_file_path, topic):
    """
    Save depth images from a ROS bag file to folder.

    Args:
        bag_file_path (str): The path to the ROS bag file.
        topic (str): The topic name to extract depth images from.

    Returns:
        None
    """

    # Create output directory
    output_dir = bag_file_path.replace('.bag', '_depth_images_32FC1/')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir) 

    # Read messages from the bag file
    with rosbag.Bag(bag_file_path, 'r') as bag:
        bridge = CvBridge()
        for _, msg, _ in bag.read_messages(topics=[topic]):
            t = msg.header.stamp
            # Convert ROS image message to OpenCV image
            image = bridge.imgmsg_to_cv2(msg, "32FC1")
            # Save the depth image
            cv2.imwrite(f"{output_dir}/depth_{t.secs}_{t.nsecs}.png", image)
