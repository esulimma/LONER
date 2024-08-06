import numpy as np
import matplotlib.pyplot as plt

PLOT = False

def main():
    """
    Generate a grid of tree coordinates and trunk radii.

    This function generates a grid of tree coordinates and trunk radii based on specified parameters.
    It uses a random seed for reproducibility and generates a list of trial (x, y) circle centers.
    The number of circles, minimum radius, maximum radius, and minimum allowable distance between centers
    can be specified. The function drops down points and saves them if they meet the minimum distance requirement.
    The function also plots the grid if the PLOT variable is set to True.

    Returns:
        coordinates (ndarray): Array of tree coordinates.
        trunk_radii (ndarray): Array of trunk radii.
    """
    
    # Initialization Steps
    np.random.seed(0)  # Seed for reproducibility

    # Get a list of trial (x, y) circle centers.
    x = 19.4 * np.random.rand(1000000)
    y = 19.4 * np.random.rand(1000000)

    # Specify how many circles are desired.
    numberOfCircles = 300
    minRadius = 0.4
    maxRadius = 0.6

    # Specify how close the centers may be to each other.
    # Should be at least twice the max radius if they are not to touch.
    minAllowableDistance = max(3, 2 * maxRadius)

    # Initialize first point.
    keeperX = [x[0]]
    keeperY = [y[0]]
    radii = [minRadius]

    # Try dropping down more points.
    counter = 1
    for k in range(1, len(x)):
        # Get a trial point.
        thisX = x[k]
        thisY = y[k]
        # See how far it is away from existing keeper points.
        distances = np.sqrt((thisX - np.array(keeperX))**2 + (thisY - np.array(keeperY))**2)
        minDistance = min(distances)
        if minDistance >= minAllowableDistance:
            # This trial location works. Save it.
            keeperX.append(thisX)
            keeperY.append(thisY)
            # Get a random radius.
            radii.append(minRadius + (maxRadius - minRadius) * np.random.rand())
            # Quit if we have enough or ran out of circles to try
            if counter >= numberOfCircles:
                break
            counter += 1

    if PLOT:
        # Plot a dot at the centers
        plt.plot(keeperX, keeperY, 'b+', markersize=9)
        for i in range(len(keeperX)):
            circle = plt.Circle((keeperX[i], keeperY[i]), radii[i], color='b', fill=False)
            plt.gca().add_artist(circle)

        plt.grid(True)
        plt.axis('equal')
        numCirclesPlaced = len(keeperX)
        caption = 'Grid with {} trees'.format(numCirclesPlaced)
        plt.title(caption, fontsize=25)
        plt.show()

    coordinates = np.transpose(np.array([keeperX, keeperY])+0.6)
    trunk_radii = np.array(radii)

    return coordinates, trunk_radii

if __name__ == "__main__":
    main()