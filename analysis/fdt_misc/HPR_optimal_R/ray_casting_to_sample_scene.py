
import open3d as o3d
import numpy as np
import scipy as sp
import os
import sys
import copy
import os
import open3d.core as o3c
from matplotlib import pyplot as plt


PLOT = False

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT.replace('/analysis',''))
sys.path.append(PROJECT_ROOT.replace('analysis', 'src'))


def get_non_detected(all_indices, detected_indices):
    # Convert arrays to sets for faster lookup
    set1 = set(all_indices)
    set2 = set(detected_indices)
    
    # Find elements only in arr1 using set difference
    non_detected_indices = set1.difference(set2)
    
    # Convert set back to a list
    return list(non_detected_indices)


def common_elements(arr1, arr2):
    # Convert arrays to sets for faster lookup
    set1 = set(arr1)
    set2 = set(arr2)
    
    # Find common elements using set intersection
    common = set1.intersection(set2)
    
    # Convert set back to a list
    return list(common)



# Function to remove hidden points from a point cloud based on a camera position and radius
def get_pcd_hidden_point_removal(pcd, camera, radius):
    """
    Raycasting to detect visible points from a point cloud based on a given camera location and HPR radius.

    Args:
        pcd (PointCloud): The input point cloud.
        camera (Camera): The camera used for hidden point removal.
        radius (float): The radius used for hidden point removal.

    Returns:
        Tuple: A tuple containing the following elements:
            - pcd_point_removal (PointCloud): The point cloud after hidden point removal.
            - number_of_points_detected (int): The number of points detected after hidden point removal.
            - pt_map (numpy.ndarray): The index map of the points after hidden point removal.
    """
    
    # Perform hidden point removal
    _, pt_map = pcd.hidden_point_removal(camera, radius)
    
    # Select points based on the index map
    pcd_point_removal = pcd.select_by_index(pt_map)
    
    # Get the number of points detected
    number_of_points_detected = len(pt_map)

    return pcd_point_removal, number_of_points_detected, pt_map

if __name__ == "__main__":
    """
    Checks HPR Performance of HPR algorithm for given camera positions and varying radii.

    Args:
        None

    Returns:
        None
    """
    red_color = [1, 0, 0]  # Red    
    blue_color = [0, 0, 1]  # Blue
    green_color = [0, 1, 0]  # Blue
    grey_color = [0.5, 0.5, 0.5]  # Blue

    current_file_path = os.path.abspath(__file__)
    render_dir = os.path.join(os.path.dirname(current_file_path), "comparison")
    arg_path = render_dir + 'sample_scene_mesh_noise' + '.ply' 
    mesh_scene_noise = o3d.io.read_triangle_mesh(arg_path)

    arg_path = render_dir + 'sample_scene_mesh_auxiliary' + '.ply'   
    mesh_scene_aux = o3d.io.read_triangle_mesh(arg_path)
    
    arg_path = render_dir + 'sample_scene_pointcloud_noise' + '.pcd'
    pcd_in = o3d.io.read_point_cloud(arg_path)


    pcd_noise = pcd_in

    points = np.asarray(pcd_noise.points)
    number_points = points.shape[0]
    center = np.array([0,0,2.5])
    vector = np.zeros(points.shape)
    centers = np.zeros(points.shape)
    ray_np = np.zeros((points.shape[0],6))

    for i in range(number_points):
        centers[i] = center
        vector[i] = (points[i] - center)/100
    
    rays_np = np.hstack((centers, vector))
    mesh_scene_aux_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_scene_aux)
    mesh_scene_noise_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_scene_noise)
    pcd_noise_t = o3d.t.geometry.PointCloud.from_legacy(pcd_noise)
    pcd_noise_t_adjusted =copy.copy(pcd_noise_t)


    rays1 = o3c.Tensor(rays_np, o3c.float32)
    scene1 = o3d.t.geometry.RaycastingScene()
    scene1.add_triangles(mesh_scene_noise_t)
    scene1.add_triangles(mesh_scene_aux_t)
    ans1 = scene1.cast_rays(rays1)

    hit1 = ans1['t_hit'].isfinite()
    points_out = rays1[hit1][:,:3] + rays1[hit1][:,3:]*ans1['t_hit'][hit1].reshape((-1,1))
    
    pcd_out = o3d.t.geometry.PointCloud(points_out)
    points_out1 = np.asarray(pcd_out.to_legacy().points)

    pcd_noise_t_adjusted.point.positions = pcd_noise_t.point.positions[hit1][:,:3]
    pcd_noise_adjusted = pcd_noise_t_adjusted.to_legacy()

    points = np.asarray(pcd_noise_adjusted.points)
    number_points = points.shape[0]

    indices_visible = np.zeros((number_points,1), dtype=int)
    indices_nonvisible = np.zeros((number_points,1), dtype=int)
    indices_all = np.linspace(0, number_points,num = number_points, endpoint = False)

    if PLOT:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        vis.get_render_option().background_color = [0, 0, 0]
        vis.add_geometry(pcd_noise.paint_uniform_color(green_color))
        vis.add_geometry(pcd_noise_adjusted.paint_uniform_color(red_color))
        # vis.add_geometry(pcd1.paint_uniform_color(green_color))
        vis.add_geometry(pcd_out.to_legacy().paint_uniform_color(blue_color))
        vis.run()

    bool_point_0_index_visible = False

    for i in range(number_points):
        if sp.spatial.distance.euclidean(points[i], points_out1[i]) < 0.11:
            indices_visible[i] = int(i)
            if (i==0):
                bool_point_0_index_visible = True
        else:
            indices_nonvisible[i] = int(i)

 
    indices_visible = indices_visible[indices_visible != 0]
    indices_nonvisible = indices_nonvisible[indices_nonvisible != 0]

    if (bool_point_0_index_visible == True):
        indices_visible = np.hstack(([0], indices_visible))
    else:
        indices_nonvisible = np.hstack(([0], indices_nonvisible))

    indices_visible_list = list(indices_visible.flatten())
    indices_nonvisible_list = list(indices_nonvisible.flatten())

    pcd_visible = pcd_noise.select_by_index(indices_visible_list)
    pcd_nonvisible = pcd_noise.select_by_index(indices_nonvisible_list)

    camera = center
    radius = 1000

    radii = np.linspace(1,10000,num=10) 
    radii = np.logspace(0,5,num=10) 
    radii = np.logspace(2,5,num=10) 

    radii = np.array([radius])
    percentage_false_positives_arr = np.zeros(radii.shape)
    percentage_false_negatives_arr = np.zeros(radii.shape)
    percentage_false_all_arr = np.zeros(radii.shape)
    hausdorff = np.zeros(radii.shape)
    chamfer = np.zeros(radii.shape)
    other_distance = np.zeros(radii.shape)

    for i in range(len(radii)):
        radius = radii[i]
        number_true = np.asarray(pcd_visible.points).shape[0]
        number_false = np.asarray(pcd_nonvisible.points).shape[0]

        # True == Visible, Positive == detected

        pcd_detected,number_positives,indices_detected_list = get_pcd_hidden_point_removal(pcd_noise,camera,radius)
        indices_nondetected_list = get_non_detected(indices_all, indices_detected_list)
        number_negatives = len(indices_nondetected_list)
        pcd_nondetected = pcd_noise.select_by_index([(int(x)) for x in indices_nondetected_list])

        true_positive_points_list = common_elements(indices_visible_list, indices_detected_list)
        false_positive_points_list = common_elements(indices_nonvisible_list, indices_detected_list)

        true_negative_points_list = common_elements(indices_nonvisible_list, indices_nondetected_list)
        false_negative_points_list = common_elements(indices_visible_list, indices_nondetected_list)

        number_true_positive_points = len(true_positive_points_list)
        number_false_positive_points = len(false_positive_points_list)

        number_true_negative_points = len(true_negative_points_list)
        number_false_negative_points = len(false_negative_points_list)

        percentage_true_positives = number_true_positive_points/number_true
        percentage_false_positives = number_false_positive_points/number_false
        percentage_true_negatives = number_true_negative_points/number_true
        percentage_false_negatives = number_false_negative_points/number_false

        percentage_true_all = (number_true_positive_points + number_true_negative_points)/number_points
        percentage_false_all = percentage_false_positives + percentage_false_negatives

        percentage_false_positives_arr[i] = percentage_false_positives
        percentage_false_negatives_arr[i] = percentage_false_negatives
        percentage_false_all_arr[i] = percentage_false_all

    index_min_percentage_false_positives =np.where(percentage_false_positives_arr == percentage_false_positives_arr.min())
    
    index_min_false_all =np.where(percentage_false_all_arr == percentage_false_all_arr.min())


    if PLOT:
        plt.semilogx(radii, percentage_false_positives_arr*100, label='false positive rate', color='green', linestyle='--', marker='o')
        plt.semilogx(radii, percentage_false_negatives_arr*100, label='false negatives rate', color='red', linestyle='--', marker='o')
        plt.semilogx(radii, percentage_false_all_arr*100, label='sum of false positive and negatives', color='blue', linestyle='--', marker='o')

        # Customizing plot
        title = 'Comparison with camera position{}'.format(camera)
        plt.xlabel('Radius [m]')
        plt.ylabel('Ratio [%]')
        plt.title(title)
        plt.legend()
        plt.grid(True)

        # Show plot
        plt.show()

    # Some assertions as a sanity check
    assertion1 = (number_positives == number_true_positive_points + number_false_positive_points)
    assertion2 = (number_negatives == number_true_negative_points + number_false_negative_points)
    assertion3 = (number_true == number_true_positive_points + number_false_negative_points)
    assertion4 = (number_false == number_false_positive_points + number_true_negative_points)
    assertion5 = (number_points == number_true_positive_points + number_false_positive_points + number_true_negative_points + number_false_negative_points)
    assertion6 = (number_points == number_positives + number_negatives)
    assertion7 = (number_points == number_true + number_false)

    assertion = assertion1 & assertion2 & assertion3 & assertion4 & assertion5 & assertion6 & assertion7
   
    # i = 0
    # for radius in radii:
    #     i = i+1
    #     pcd_hidden,_,indices = get_pcd_hidden_point_removal(pcd_noise,camera,radius)
    #     false_pos[i] = 1
    #     false_neg[i] = 0

    if PLOT:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        vis.get_render_option().background_color = [0, 0, 0]
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0,origin =center).paint_uniform_color(grey_color))
        # vis.add_geometry(box_mesh)
        # vis.add_geometry(pcd1.paint_uniform_color(green_color))    
        vis.add_geometry(pcd_visible.paint_uniform_color(red_color))
        vis.add_geometry(pcd_nonvisible.paint_uniform_color(grey_color))
        vis.run()

    if PLOT:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        vis.get_render_option().background_color = [0, 0, 0]
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0,origin =center).paint_uniform_color(green_color))
        vis.add_geometry(pcd_detected.paint_uniform_color(green_color))
        vis.add_geometry(pcd_nondetected.paint_uniform_color(blue_color))
        vis.run()
