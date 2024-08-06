#!/usr/bin/env python
# coding: utf-8

import os
import sys
import random
import math
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT.replace('/analysis',''))
sys.path.append(PROJECT_ROOT.replace('analysis', 'src'))

GRID_SIZE = 20
MIN_DISTANCE = 1.5

def distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def generate_point(existing_points):
    """Generate a new random point."""
    while True:
        x = random.uniform(0, GRID_SIZE)
        y = random.uniform(0, GRID_SIZE)
        valid = True
        for point in existing_points:
            if distance((x, y), point) < MIN_DISTANCE:
                valid = False
                break
        if valid:
            return (x, y)

def plot_grid_with_points(grid, points):
    """Plot the grid with points."""
    plt.figure(figsize=(8, 8))
    
    # Plot grid lines
    for i in range(GRID_SIZE + 1):
        plt.plot([i, i], [0, GRID_SIZE], color='k', linewidth=0.5)
        plt.plot([0, GRID_SIZE], [i, i], color='k', linewidth=0.5)
    
    # Plot points
    for point in points:
        plt.scatter(point[0], point[1], color='red')
    
    
    plt.xlim(0, GRID_SIZE)
    plt.ylim(0, GRID_SIZE)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Grid with Random Points')
    plt.grid(True)
    plt.show()

def main():
    """
    Generates a grid and plots it with randomly generated points.
    """
    
    grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
    points = []
    
    for _ in range(GRID_SIZE * GRID_SIZE):
        new_point = generate_point(points)
        points.append(new_point)
        grid[int(new_point[0])][int(new_point[1])] = 1
    
    plot_grid_with_points(grid, points)

if __name__ == "__main__":
    main()
