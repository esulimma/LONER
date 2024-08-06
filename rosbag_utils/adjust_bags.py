import os
import shutil
import rosbag
import ros_numpy
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage

from extract_scans import remove_box_from_point_cloud
from extract_calibration import get_calibration
from extract_images import save_compressed_images
from image_detect_keypoints import lucas_canade
from image_segmentation import vit_h
from extract_trajectories import extract_gt_o3d_tum

def adjust_and_extract_all(input_bag_path, input_gt_bag_path):
    """
    Adjusts the input bag file and extracts calibration and ground truth trajectory.

    Parameters:
    - input_bag_path (str): The path to the input bag file.
    - input_gt_bag_path (str): The path to the ground truth bag file.

    Returns:
        None
    """

    # Get calibration parameters from the input bag file
    get_calibration(input_bag_path)

    # Adjust the input bag file
    adjust_bag(input_bag_path)

    # Extract ground truth data from the ground truth bag file
    extract_gt_o3d_tum(input_gt_bag_path)

def adjust_bag(input_bag_path):
    """
    Adjusts a bag file by cloning it, reading and adjusting scans, and creating and inserting masks.

    Args:
        input_bag_path (str): The path to the input bag file.

    Returns:
        None
    """
    images_folder = os.path.dirname(input_bag_path) + '/images/'
    
    # Clone bag and adjust scans in the cloned bag file
    adjusted_input_bag_path = read_and_adjust_scans(input_bag_path)

    # Save compressed images from the adjusted bag file
    save_compressed_images(adjusted_input_bag_path)

    # Apply Lucas-Canade algorithm to the images folder
    lucas_canade(images_folder, save_images=True)

    # Apply Vit-H algorithm to the images folder
    vit_h(images_folder, save_images=True)

    # Read and insert masks in the cloned bag file
    read_and_insert_masks(adjusted_input_bag_path)


def clone_bag(input_bag_path):
    """
    Clones a ROS bag file by creating a copy with '_adjusted' suffix in the same location.

    Parameters:
    - input_bag_path: File path pointing to the input bag file to be cloned.
    Returns:
        str: The file path of the cloned bag file.
    """
    output_bag_path = input_bag_path.replace('.bag', '_adjusted.bag')
    shutil.copyfile(input_bag_path, output_bag_path)
    print(f"Bag {input_bag_path} cloned to {output_bag_path}.")
    return output_bag_path

def append_bags(output_bag_path, input_bag_paths):
    """
    Appends multiple ROS bag files into a single output bag file.

    Parameters:
    - output_bag_path (str): The file path where the output bag file will be saved.
    - input_bag_paths (list of str): A list of file paths pointing to the input bag files to be appended.
    Returns:
        None
    """
    print(f"Appending {len(input_bag_paths)} rosbags.")
    with rosbag.Bag(output_bag_path, 'w') as outbag:
        for bag_file in input_bag_paths:
            with rosbag.Bag(bag_file) as inbag:
                for topic, msg, t in inbag.read_messages():
                    outbag.write(topic, msg, t)
    print("Bags appended successfully to:")
    print(output_bag_path)


def read_and_adjust_scans(input_bag_file, box_dimensions = (4.5, 7, 60), 
                          box_origin = (0, 0.5, 2.5), lidar_topic = '/hesai/pandar'):
    # http://wiki.ros.org/rosbag/Cookbook#Rewrite_bag_with_header_timestamps
    """
    Read a bag file containing a lidar point cloud, remove points within a specified box,
    when self_removal node is not available.

    Args:
    - input_bag_file: path to the input bag file
    - box_dimensions: tuple (length, width, height) of the box dimensions in meters
    - box_origin: tuple (x, y, z) representing the center of the box in lidar coordinates
    - lidar_topic
    Returns:
        None
    """
    inbag = rosbag.Bag(input_bag_file)
    output_bag_file = input_bag_file.replace('.bag', '_adjusted.bag')  # Replace with the desired path for the output bag file
    print("---------------------------------------------------------------------")
    print(f"Adjusting scans in {input_bag_file} and saving to {output_bag_file}")
    print(f"With ROS lidar topic: {lidar_topic}")
    counter = 0
    with rosbag.Bag(output_bag_file, 'w') as outbag:
        for _, msg, _ in inbag.read_messages(topics=[lidar_topic]):
                points = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
                adjusted_points = remove_box_from_point_cloud(points, box_dimensions, box_origin)
                adjusted_pc2_msg = ros_numpy.point_cloud2.array_to_pointcloud2(adjusted_points, msg.header.stamp, msg.header.frame_id)
                outbag.write(lidar_topic, adjusted_pc2_msg, msg.header.stamp)
                counter += 1
        for topic, msg, t in inbag.read_messages():
            if topic == lidar_topic:
                continue
            else:
                outbag.write(topic, msg, msg.header.stamp if msg._has_header else t)
    inbag.close()
    outbag.close()
    print(f"Adjusted {counter} scans from {input_bag_file}.")
    return output_bag_file

def read_and_insert_masks(input_bag_file, 
                          image_topic='/zed2i/zed_node/rgb/image_rect_color/compressed',
                          mask_topic='/zed2i/zed_node/rgb/image_rect_color/mask'):
    """
    Reads images from a ROS bag file, inserts corresponding masks, and writes them back to the bag file.

    Args:
        input_bag_file (str): The path to the input ROS bag file.
        image_topic (str, optional): The topic name for the input images. Defaults to '/zed2i/zed_node/rgb/image_rect_color/compressed'.
        mask_topic (str, optional): The topic name for the output masks. Defaults to '/zed2i/zed_node/rgb/image_rect_color/mask'.
    Returns:
        None
    """
    image_masks_path = os.path.dirname(input_bag_file) + '/' +  'image_masks/'
    # bridge = CvBridge()

    with rosbag.Bag(input_bag_file, 'a') as bag:
        for _, msg, _ in bag.read_messages(topics=[image_topic]):
            # Load mask corrsponding to timestamp
            t = msg.header.stamp
            n_str = str(t.nsecs)
            if len(n_str) < 9:
                n_str = n_str.zfill(9)
            name = f"/image_{t.secs}_" + n_str + ".npy"
            mask = np.load(image_masks_path + name).astype('uint8')
            # Convert mask to ROS image message
            # ros_image = bridge.cv2_to_imgmsg(mask, encoding="mono8")
            # ros_image.header.stamp = msg.header.stamp
            compressed_image = CompressedImage()
            compressed_image.header.stamp = msg.header.stamp
            compressed_image.format = "jpeg"
            _, buffer = cv2.imencode('.jpg', mask)
            compressed_image.data = buffer.tobytes()
            # Write mask to bag file
            bag.write(mask_topic, compressed_image, msg.header.stamp)
    bag.close()