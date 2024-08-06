#!/usr/bin/env python
# coding: utf-8

import open3d as o3d
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT.replace('/analysis',''))
sys.path.append(PROJECT_ROOT.replace('analysis', 'src'))

def generate_random_point_in_cylinder(radius, height):
    """
    Generate a random point within a cylinder.

    Args:
        radius (float): Radius of the cylinder.
        height (float): Height of the cylinder.

    Returns:
        np.array: Random point in Cartesian coordinates [x, y, z].
    """
    # Generate random points in cylindrical coordinates
    r = np.random.uniform(0, radius)
    theta = np.random.uniform(0, 2 * np.pi)
    z = np.random.uniform(0, height)
    
    # Convert cylindrical coordinates to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return np.array([x, y, z])

def density_function(point, radius):
    """
    Calculate the density of a point based on its distance from the center.

    Args:
        point (np.array): Point in Cartesian coordinates [x, y, z].
        radius (float): Radius of the density function.

    Returns:
        float: Density value.
    """
    # Calculate distance from center
    distance = np.linalg.norm(point[:2])
    # Calculate density based on distance
    density = np.exp(-distance / radius)
    return density

def generate_random_points_with_density(num_points, radius, height, min_distance):
    """
    Generate random points within a cylinder with varying densities.

    Args:
        num_points (int): Number of points to generate.
        radius (float): Radius of the cylinder.
        height (float): Height of the cylinder.
        min_distance (float): Minimum distance from the center to remove trunk points.

    Returns:
        tuple: Tuple containing the generated points and their corresponding densities.
    """
    points = []
    max_density = 0
    for _ in range(num_points):
        point = generate_random_point_in_cylinder(radius, height)
        density = density_function(point, radius)
        max_density = max(max_density, density)
        points.append(point)
    
    # Normalize densities
    points = np.array(points)
    densities = np.array([density_function(point, radius) / max_density for point in points])

    # Remove Trunk Points
    center_distance = np.linalg.norm(points[:, :2], axis=1)
    points = points[center_distance > min_distance]
    densities = densities[center_distance > min_distance]
    
    return points, densities

def main(branch_radius, trunc_radius, num_points, height):
    """
    Main function to generate random points within a cylinder and create a point cloud.

    Args:
        branch_radius (float): Radius of the cylinder.
        trunc_radius (float): Minimum distance from the center to remove trunk points.
        num_points (int): Number of points to generate.
        height (float): Height of the cylinder.

    Returns:
        o3d.geometry.PointCloud: Generated point cloud.
    """
    points, _ = generate_random_points_with_density(num_points, branch_radius, height, trunc_radius)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    return pcd 

if __name__ == "__main__":
    main()
