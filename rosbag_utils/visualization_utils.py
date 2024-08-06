import os
import numpy as np
import glob
from PIL import Image
import open3d as o3d

def fade(shape, fimage1, fimage2, perc):
    """
    Fades two images horizontally based on a given percentage.

    Parameters:
    - shape (tuple): The shape of the output image.
    - fimage1 (ndarray): The first image to fade.
    - fimage2 (ndarray): The second image to fade.
    - perc (float): The percentage of the first image to be shown.

    Returns:
    - ndarray: The faded image.
    """

    # Calculate the index to split the images based on the percentage
    i = int(fimage1.shape[1] * perc) + 1

    # Concatenate the two images horizontally and reshape to the desired shape
    return np.concatenate((fimage2[:, :i], fimage1[:, i:]), axis=1).reshape(shape)


def split_image_gif(image_1_path, image_2_path, output_folder):
    """
    Creates a GIF by splitting two images and fading between them.

    Parameters:
    image_1_path (str): The file path of the first image.
    image_2_path (str): The file path of the second image.
    output_folder (str): The folder where the GIF will be saved.

    Returns:
    None
    """

    # Read the images
    image1 = np.array(Image.open(image_1_path))
    image2 = np.array(Image.open(image_2_path))
    
    # Get the width of the second image
    width = image2.shape[1]

    # Create frames for the GIF
    frames = []
    for i in range(0, width, 2):
        new_image = fade(image1.shape, image1, image2, i/100)
        frames.append(Image.fromarray(new_image))

    # Reverse the frames and append them to the original frames
    new_frames = frames.copy()
    for i in range(len(frames)):
        new_frames.append(frames[len(frames)-i-1])

    # Save the GIF
    new_frames[0].save(output_folder+'00.gif', 
                save_all=True, append_images=new_frames[1:], 
                optimize=False, loop=0)

def render_gif(frame_folder, sorting='time', range=None, skip_step=1):
    """
    Renders a GIF animation from a folder containing image frames.

    Args:
        frame_folder (str): The path to the folder containing the image frames.
        sorting (str, optional): The sorting method for the image frames. 
            Valid options are 'time', 'name', 'number', and 'default'. 
            Defaults to 'time'.
        range (list, optional): The range of frames to include in the GIF. 
            Defaults to None, which includes all frames.
        skip_step (int, optional): The step size for skipping frames. 
            Defaults to 1, which includes every frame.

    Raises:
        AssertionError: If an invalid sorting method is provided.
        AssertionError: If no images are found in the folder.

    Returns:
        None
    """

    # Get a list of all image files in the folder
    files = list(filter(os.path.isfile, glob.glob(frame_folder + "*.jpg")))

    # If no JPG files found, try finding PNG files
    if files == []:
        files = list(filter(os.path.isfile, glob.glob(frame_folder + "*.png")))

    # Sort the files based on the specified sorting method
    if sorting == 'time':
        files.sort(key=lambda x: os.path.getmtime(x))
    elif sorting == 'name':
        files.sort(key=lambda x: os.path.basename(x))
    elif sorting == 'number':
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    elif sorting == 'default':
        files.sort()
    else:
        assert False, "Invalid sorting method"

    # Set the range of frames to include in the GIF
    if range is None:
        range = [0, len(files)-1]
    start = range[0]
    end = range[1]

    # Load the image frames
    frames = []
    for file_name in files[start:skip_step:end]:
        frames.append(Image.open(file_name))

    # Check if any images were found in the folder
    try:
        frame_one = frames[0]
    except:
        assert False, "No images found in the folder"

    # Print the filenames of the frames
    for frame in frames:
        print(frame.filename)

    # Save the GIF animation
    frame_one.save(frame_folder+'00.gif', 
                   save_all=True, append_images=frames[1:], 
                   optimize=False, loop=0)


def convert_ply(ply_path, plot=False):
    """
    Converts a PLY file to a PCD file using Open3D library.

    Args:
        ply_path (str): The path to the PLY file.
        plot (bool, optional): Whether to plot the point cloud with a coordinate frame. Defaults to False.

    Returns:
        None
    """

    pcd = o3d.io.read_point_cloud(ply_path)
    outpath = ply_path.replace('.ply', '.pcd')

    # Add coordinate frame
    if plot:
        aabb = pcd.get_axis_aligned_bounding_box()
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        vis.get_render_option().background_color = [0, 0, 1]
        vis.add_geometry(pcd.paint_uniform_color([1, 0, 0]))
        vis.add_geometry(aabb)
        vis.add_geometry(coord_frame)
        vis.run()

    o3d.io.write_point_cloud(outpath, pcd)
