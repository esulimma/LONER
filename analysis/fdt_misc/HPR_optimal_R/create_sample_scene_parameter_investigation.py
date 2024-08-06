#!/usr/bin/env python
# coding: utf-8

import open3d as o3d
import os
import sys
import numpy as np
import copy
import random
import matplotlib.pyplot as plt

import create_sampling_noise
import create_tree_grid

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.dirname(PROJECT_ROOT))

from fdt_common_utils.metrics import get_pcd_distances

def main(): 
    """
    This function generates a sample scene for parameter investigation.
    It creates a scene with trees and a ground surface, adds noise to the trees and the ground,
    and calculates distances between the noisy and noise-free point clouds.
    The distances are then plotted against the noise level.
    """

    main_grid_size = 30
    trees_grid_size =20
    tree_coordinates,trunc_radii = create_tree_grid.main()

    mesh_trees = o3d.geometry.TriangleMesh()

    branch_radius = 2.0
    trunc_radius = 0.5 
    num_points = 100 
    branch_height = 5

    pcd_branch = create_sampling_noise.main(branch_radius, trunc_radius, num_points, branch_height)

    pcd_branch_noise = o3d.geometry.PointCloud()

    num_points = np.linspace(10,100,num = 9)    

    for i in range(len(trunc_radii)):
        tree_height = 12 + random.uniform(-3, 3)
        branch_height = tree_height * 0.25
        trunc_radius = trunc_radii[i]

        mesh_tree = o3d.geometry.TriangleMesh.create_cylinder(radius=trunc_radius, height=tree_height, resolution=20, split=4)
        pcd_branch = create_sampling_noise.main(branch_radius, trunc_radius, num_points, branch_height)

        translation = [tree_coordinates[i,0],tree_coordinates[i,1],tree_height/2]
        mesh_tree.translate(translation)
        translation = [tree_coordinates[i,0],tree_coordinates[i,1],tree_height-branch_height]
        pcd_branch.translate(translation)

        mesh_trees = mesh_trees + mesh_tree
        pcd_branch_noise = pcd_branch_noise + pcd_branch


    d_grid_compensation = (main_grid_size - trees_grid_size)/2
    translation_grid_compensation = [d_grid_compensation,d_grid_compensation,0]
    mesh_trees.translate(translation_grid_compensation)
    pcd_branch_noise.translate(translation_grid_compensation)

    side_length = main_grid_size

    # Create Ground Surface from Box
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=side_length, height=side_length, depth=side_length)
    mesh_ground = o3d.geometry.TriangleMesh()
    mesh_ground.vertices = mesh_box.vertices
    mesh_ground.triangles = o3d.utility.Vector3iVector(np.asarray(mesh_box.triangles)[8:10])
    mesh_ground.translate([0, 0 ,-side_length])

    mesh_scene = mesh_ground + mesh_trees
    
    number_sample_points_tree = 750
    number_sample_points_ground = 750

    number_sample_points = number_sample_points_tree + number_sample_points_ground

    pcd = mesh_scene.sample_points_poisson_disk(number_sample_points)
    pcd.paint_uniform_color([1,0,0])
   
    pcd_trees_no_noise = mesh_trees.sample_points_poisson_disk(number_sample_points_tree)
    pcd_ground_no_noise = mesh_ground.sample_points_poisson_disk(number_sample_points_ground)
    pcd_no_noise = pcd_trees_no_noise + pcd_ground_no_noise

    sigmas = np.linspace(0.05,0.4,7,endpoint=False)

    distances_tree_to_tree = np.zeros((7,3))

    for i in range(len(sigmas)):
        mu = 0
        sigma = sigmas[i]
        mesh_ground_noise = copy.copy(mesh_ground)
        mesh_trees_noise = copy.copy(mesh_trees)
        mesh_trees_noise = mesh_trees_noise.subdivide_loop(number_of_iterations=2)
        vertices = np.asarray(mesh_trees_noise.vertices)  
        mesh_trees_noise.vertices = o3d.utility.Vector3dVector(vertices + np.random.uniform(mu, sigma, size=vertices.shape))      

        pcd_sensor_noise_trees = mesh_trees_noise.sample_points_poisson_disk(number_sample_points_tree)
        pcd_sensor_noise_ground = mesh_ground_noise.sample_points_poisson_disk(number_sample_points_ground)
        pcd_sensor_noise = pcd_sensor_noise_trees + pcd_sensor_noise_ground
        pcd_noise = pcd_branch_noise + pcd_sensor_noise

        distances_tree_to_tree[i] = get_pcd_distances.get_distances(pcd_noise,pcd_no_noise)


    # Plotting
    plt.plot(sigmas, distances_tree_to_tree[:,0], label={'Chamfer'}, color='green', linestyle='--', marker='o')
    plt.plot(sigmas, distances_tree_to_tree[:,1], label={'Earth Mover'}, color='blue', linestyle='--', marker='o')
    plt.plot(sigmas, distances_tree_to_tree[:,2], label={'Hausdorff'}, color='red', linestyle='--', marker='o')
    # plt.semilogx(radii, percentage_false_all_arr*100, label='sum of false positive and negatives', color='blue', linestyle='--', marker='o')

    # Customizing plot
    title = 'comparison vertex noise on point cloud distances'
    plt.xlabel('Sigma')
    plt.ylabel('Distances')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()

    return 


if __name__ == "__main__":
    main()