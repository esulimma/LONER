import os
import glob
import rosbag
import ros_numpy
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def remove_box_from_point_cloud(point_cloud, box_dimensions, box_origin):
    """
    Remove points within a bounding box from a lidar point cloud when self_removal node is not available

    Args:
    - point_cloud: numpy array of shape (N, 3) containing lidar points (x, y, z)
    - box_dimensions: tuple (length, width, height) of the box dimensions in meters
    - box_origin: tuple (x, y, z) representing the center of the box in lidar coordinates

    Returns:
    - filtered_point_cloud (numpy array): point cloud with points inside the box removed
    """

    length, width, height = box_dimensions
    box_x, box_y, box_z = box_origin

    # Define boundaries of the box
    min_x = box_x - length / 2
    max_x = box_x + length / 2
    min_y = box_y - width / 2
    max_y = box_y + width / 2
    min_z = box_z - height / 2
    max_z = box_z + height / 2

    # Find points outside the box
    mask = ((point_cloud['x'] < min_x) | (point_cloud['x'] > max_x) |
        (point_cloud['y'] < min_y) | (point_cloud['y'] > max_y) |
        (point_cloud['z'] < min_z) | (point_cloud['z'] > max_z))

    filtered_point_cloud = point_cloud[mask]

    return filtered_point_cloud

def crop_and_save_scan_as_image(point_cloud, output_dir, index, plot_inside=True):
    """
    Crop the point cloud based on specified limits and save the resulting image.

    Args:
        point_cloud (dict): Dictionary containing the point cloud data with keys 'x', 'y', and 'z'.
        output_dir (str): Directory path where the image will be saved.
        index (int): Index of the image.
        plot_inside (bool, optional): Whether to plot the points inside the specified limits. 
            Defaults to True.

    Returns:
        None
    """
    # Set the limits for cropping
    min_x = -15
    max_x = 15
    min_y = -15
    max_y = 15
    min_z = -4
    max_z = 12

    # Create masks for points outside and inside the limits
    mask_inside = ((point_cloud['x'] < min_x) | (point_cloud['x'] > max_x) |
                   (point_cloud['y'] < min_y) | (point_cloud['y'] > max_y) |
                   (point_cloud['z'] < min_z) | (point_cloud['z'] > max_z))

    mask = ((point_cloud['x'] < max_x) & (point_cloud['x'] > min_x) &
            (point_cloud['y'] < max_y) & (point_cloud['y'] > min_y) &
            (point_cloud['z'] < max_z) & (point_cloud['z'] > min_z))

    # Apply the masks to filter the point cloud
    filtered_point_cloud_inside = point_cloud[mask_inside]
    filtered_point_cloud = point_cloud[mask]

    # Extract the coordinates of the filtered points
    points = np.zeros((len(filtered_point_cloud['x']), 3))
    points_inside = np.zeros((len(filtered_point_cloud['x']), 3))

    points[:, 0], points[:, 1], points[:, 2] = [filtered_point_cloud['x'], filtered_point_cloud['y'], filtered_point_cloud['z']]
    points_inside[:, 0], points_inside[:, 1], points_inside[:, 2] = [filtered_point_cloud_inside['x'], filtered_point_cloud_inside['y'], filtered_point_cloud_inside['z']]

    # Plot the points
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=90)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.1, c='r')
    if plot_inside:
        ax.scatter(points_inside[:, 0], points_inside[:, 1], points_inside[:, 2], s=0.1, c='grey')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-1, 10])
    # plt.show()

    # Save the image
    plt.savefig(os.path.join(output_dir, f'image_{index}.png'), bbox_inches='tight', dpi=150)
    plt.close()


def read_rosbag_and_save_scan_images(bag_file_path, lidar_topic = '/hesai/pandar', skip_step = 10):
    """
    Read a ROS bag file and save images of point clouds from the specified topic.
    """
    output_dir = bag_file_path.replace('.bag', '_pcd_images_rosbag/')

    if not(os.path.isdir(output_dir)):
        os.mkdir(output_dir) 

    i = 0
    with rosbag.Bag(bag_file_path, 'r') as bag:
        for _, msg, t in bag.read_messages(topics = [lidar_topic]):
            if i == skip_step:
                time_stamp = f"{t.secs}_{t.nsecs}"
                point_cloud = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
                crop_and_save_scan_as_image(point_cloud, output_dir, time_stamp)
                i = 1
            else:
                i = i + 1


def crop_and_save_pcd_as_image(point_cloud, output_dir, index):
    """
    Crop the given point cloud based on specified limits and save it as an image.

    Args:
        point_cloud (open3d.geometry.PointCloud): The input point cloud.
        output_dir (str): The directory to save the image.
        index (int): The index of the image.

    Returns:
        None
    """
    # Set the limits for cropping
    min_x = -15; max_x = 15
    min_y = -15; max_y = 15
    min_z = -4; max_z = 12

    # Create a mask to filter points within the limits
    points = np.asarray(o3d.utility.Vector3dVector(point_cloud.points)).T
    mask = ((points[0]  < max_x) & (points[0] > min_x) &
            (points[1]  < max_y) & (points[1] > min_y) &
            (points[2]  < max_z) & (points[2]  > min_z))

    # Apply the mask to the point cloud
    filtered_point_cloud = point_cloud.select_by_index(np.where(mask)[0])
    points = np.asarray(o3d.utility.Vector3dVector(filtered_point_cloud.points))

    # Plot the points
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=90)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_zlim([-1, 10])
    # plt.show()

    # Save the image
    plt.savefig(os.path.join(output_dir, f'image_{index}.png'), bbox_inches='tight', dpi=150)
    plt.close()


def read_pcds_and_save_images(pcd_file_path, skip_step=1):
    """
    Reads a collection of .pcd files from the given directory path and saves them as images.

    Args:
        pcd_file_path (str): The path to the directory containing the .pcd files.
        skip_step (int, optional): The step size for skipping files. Defaults to 1.

    Returns:
        None
    """
    # Create the output directory if it doesn't exist
    output_dir = pcd_file_path.replace('pcd_renders', '_pcd_images_pcd')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir) 

    # Get a list of .pcd files in the directory and sort them by modification time
    files = list(filter(os.path.isfile, glob.glob(pcd_file_path + "*.pcd")))
    files.sort(key=lambda x: os.path.getmtime(x))   

    # Iterate over the files and process each .pcd file
    for file_index, file_name in enumerate(files[::skip_step]):
        if file_name.endswith('.pcd'):
            point_cloud = o3d.io.read_point_cloud(file_name)
            crop_and_save_pcd_as_image(point_cloud, output_dir, file_index)
