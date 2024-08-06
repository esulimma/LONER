import cv2
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)

data_path = os.path.dirname(PROJECT_ROOT) + '/data'

def remove_object(images_folder, output_folder):
    """
    Remove objects from images using background subtraction.

    Args:
        images_folder (str): Path to the folder containing the input images.
        output_folder (str): Path to the folder where the processed images will be saved.

    Returns:
        None
    """
    
    # Create background subtractor object
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    for filename in os.listdir(images_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(images_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Unable to load image {filename}")
                continue

            # Apply background subtraction
            fg_mask = bg_subtractor.apply(image)

            # Invert the mask to get the background
            bg_mask = cv2.bitwise_not(fg_mask)

            # Apply the mask to the image
            image_removed_object = cv2.bitwise_and(image, image, mask=bg_mask)

            # Save the processed image
            cv2.imwrite(output_path, image_removed_object)
            print(f"Object removed from {filename}")

def watershed(images_folder, output_folder):
    """
    Applies the watershed algorithm to each image in the given folder.

    Args:
        images_folder (str): The path to the folder containing the input images.
        output_folder (str): The path to the folder where the processed images will be saved.
    """

    for filename in os.listdir(images_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(images_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Unable to load image {filename}")
                continue

            # Apply watershed algorithm
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            sure_bg = cv2.dilate(thresh, None, iterations=3)
            sure_fg = cv2.erode(thresh, None, iterations=2)
            unknown = cv2.subtract(sure_bg, sure_fg)
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(image, markers)
            image_with_markers = image.copy()
            image_with_markers[markers == -1] = [255, 0, 0]  # Outline the watershed lines

            # Save the processed image
            cv2.imwrite(output_path, image_with_markers)
            print(f"Watershed applied to {filename}")

def canny(images_folder, output_folder):
    """
    Applies Canny edge detection to images in the specified folder.

    Args:
        images_folder (str): The path to the folder containing the input images.
        output_folder (str): The path to the folder where the output images will be saved.

    Returns:
        None
    """

    # Iterate over the files in the images folder
    for _, filename in enumerate(os.listdir(images_folder)[0:1]):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(images_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Load the image
            image = cv2.imread(image_path)
                        
            if image is None:
                print(f"Error: Unable to load image {filename}")
                continue

            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to the grayscale image
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply Canny edge detection to the blurred image
            edges1 = cv2.Canny(blur, 10, 400)

            # Apply bilateral filter to the original image
            blur = cv2.bilateralFilter(image, 20, 45, 275)

            # Apply Canny edge detection to the filtered image
            edges = cv2.Canny(blur, threshold1=180, threshold2=200, apertureSize=5, L2gradient=True)

            # Apply a 1D filter to the edges
            kernel = np.ones((10, 1), np.float32) / 10
            dst = cv2.filter2D(edges, -1, kernel)
            cv2.imwrite(output_path, dst)

            # Calculate the difference between the two edge images
            delta = edges - edges1
            delta[delta < 0] = 0

            cv2.imwrite(output_path.replace('image_','delta_edges_'), delta)
            print(f"Canny Edge Detection applied to {filename}")

def harris(images_folder, output_folder):
    """
    Apply Harris Corner Detection to images in a folder and save the output images with detected corners.

    Args:
        images_folder (str): Path to the folder containing the input images.
        output_folder (str): Path to the folder where the output images will be saved.

    Returns:
        None
    """

    # Iterate over the files in the images folder
    for filename in os.listdir(images_folder)[0:1]:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(images_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Unable to load image {filename}")
                continue

            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply Harris Corner Detector
            dst = cv2.cornerHarris(gray, blockSize=25, ksize=25, k=0.04)

            # Dilate the corner points to make them more visible
            dst = cv2.dilate(dst, None)

            # Threshold for an optimal value, it may vary depending on the image
            image[dst > 0.01 * dst.max()] = [0, 0, 255]  # Mark detected corners in red

            # Save the output image with detected corners
            cv2.imwrite(output_path.replace('image_','image_harris_'), image)
            print(f"Harris Corner Detection applied to {filename}")

def average(images_folder, output_folder):
    """
    Apply average detection to a set of images and save the output image with detected corners.

    Args:
        images_folder (str): The path to the folder containing the input images.
        output_folder (str): The path to the folder where the output image will be saved.

    Returns:
        None
    """

    output_path = os.path.join(output_folder, os.listdir(images_folder)[0])

    # Assuming image_path1, image_path2, ..., image_path8 are the paths to your images
    image_paths = os.listdir(images_folder)[0:8]

    # Load images and convert them to NumPy arrays
    images = [cv2.imread(os.path.join(images_folder,path)) for path in image_paths]

    # Convert images to float32 for averaging
    images_float32 = [np.float32(img) for img in images]

    # Average the images
    average_image = np.mean(images_float32, axis=0)
    average_image = np.uint8(average_image)  

    # Save the output image with detected corners
    cv2.imwrite(output_path.replace('image_','image_average_'), average_image)
    print(f"Average Detection applied to")

def SIFT(images_folder, output_folder):
    """
    Perform SIFT (Scale-Invariant Feature Transform) on a set of images.

    Args:
        images_folder (str): Path to the folder containing the input images.
        output_folder (str): Path to the folder where the output images will be saved.

    Returns:
        None
    """

    # Loop through the images in the folder
    for i, filename in enumerate(os.listdir(images_folder)[0:1]):
        # Initialize SIFT detector
        sift = cv2.SIFT_create()
        keypoints_list = []
        descriptors_list = []

        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(images_folder, filename)
            output_path = os.path.join(output_folder, filename)

            image_path = os.path.join(images_folder, "00_reference_image.png")

            # Load the image
            image = cv2.imread(image_path)
            image2 = cv2.imread(os.path.join(images_folder, os.listdir(images_folder)[i+1]))

            keypoints, descriptors = sift.detectAndCompute(image, None)
            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)
            keypoints, descriptors = sift.detectAndCompute(image2, None)
            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)

            # Feature matching
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors_list[0], descriptors_list[1], k=2)  # Change index for different image pairs

            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append([m])
                else:
                    good_matches.append([m])

            # Draw matches
            matched_img = cv2.drawMatchesKnn(image, keypoints_list[0], image2, keypoints_list[1], good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Display matched image
            # cv2.imshow("Matched Image", matched_img)

            cv2.imwrite(output_path.replace('image_','image_matched_'), matched_img)


def crop_image(image, range=[80, 380]):
    """
    Crop the input image based on the specified range.

    Parameters:
    image (numpy.ndarray): The input image to be cropped.
    range (list, optional): The range of columns to keep in the cropped image. Defaults to [80, 380].

    Returns:
    numpy.ndarray: The cropped image.
    """
    cropped_image = image[:, range[0]:range[1]]
    return cropped_image

def crop_points(points, range=[80, 380]):
    """
    Crop the points based on a given range.

    Args:
        points (numpy.ndarray): The input array of points.
        range (list, optional): The range to crop the points. Defaults to [80, 380].

    Returns:
        numpy.ndarray: The cropped points array.
    """
    # Crop the points by subtracting the lower range value from the x-coordinate
    cropped_points = points
    cropped_points[:, 0] = points[:, 0] - range[0]

    # Filter out the points that are outside the range
    return cropped_points[cropped_points[:, 0] < range[1]]


def uncrop_points(points, range=[80, 380]):
    """
    Uncrops the given points by adding the specified range to the x-coordinate and filtering out points outside the range.

    Args:
        points (numpy.ndarray): The input points to be uncropped.
        range (list, optional): The range to be added to the x-coordinate. Defaults to [80, 380].

    Returns:
        numpy.ndarray: The uncropped points.

    """
    # Add the range to the x-coordinate
    uncropped_points = points
    uncropped_points[:, 0] = points[:, 0] + range[0]

    # Filter out points outside the range
    return uncropped_points[uncropped_points[:, 0] < range[1]]

def lucas_canade(images_folder, reference_mask="00_reference_mask.png", save_images=False):
    """
    Apply Lucas-Kanade optical flow algorithm to track keypoints in a sequence of images.

    Args:
        images_folder (str): Path to the folder containing the input images.
        reference_mask (str, optional): Name of the reference mask image file. Defaults to "00_reference_mask.png".
        save_images (bool, optional): Whether to save the result images with tracked keypoints. Defaults to False.
    """

    # Create output folders
    output_folder = images_folder.replace('images', 'images_tracked') + "/"
    os.makedirs(output_folder, exist_ok=True)

    output_folder_np = images_folder.replace('images', 'image_keypoints') + "/"
    os.makedirs(output_folder_np, exist_ok=True)

    print("---------------------------------------------------------------------")
    print(f"Applying Lucas-Kanade Optical Flow Algorithm to Track Keypoints from {images_folder}")

    # Load the first image
    os.chdir(images_folder)
    image_path = os.path.join(images_folder, os.listdir(images_folder)[0])
    old_frame = cv2.imread(image_path)

    # Keypoints to track
    p0 = np.array([
        [302, 309], [332, 354], [289, 316], [285, 293], [154, 282],
        [136, 316], [124, 338], [192, 329], [205, 340], [202, 292],
        [188, 222], [236, 176], [185, 121], [204, 67], [176, 12],
        [196, 20], [170, 94], [168, 129], [166, 162], [181, 218],
        [178, 250], [178, 277], [198, 327], [291, 320], [123, 355],
        [231, 147], [220, 185], [184, 160], [169, 116], [203, 74],
        [184, 22]
    ])

    p0_cropped = crop_points(p0)
    p0_cropped = p0_cropped.reshape(-1, 1, 2)
    p0_cropped = p0_cropped.astype(np.float32)

    # Load the reference mask
    mask_path = os.path.join(os.path.dirname(images_folder), reference_mask)

    if not os.path.exists(mask_path):
        mask_path = os.path.join(os.path.dirname(os.path.dirname(images_folder)), reference_mask)

    if os.path.exists(mask_path):
        print(f"Reference mask found at {mask_path}")
        mask = cv2.imread(mask_path)

        mask_adjusted = mask[:, :, 0] + mask[:, :, 1] + mask[:, :, 2]
        mask_adjusted[mask_adjusted > 0] = 1
        mask_adjusted_cropped = crop_image(mask_adjusted)
    else:
        print(f"Reference mask not found at {mask_path}")
        print(f"Add Mask in Order to Improve Performance")
        mask_adjusted_cropped = None

    # Convert the first frame to grayscale
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    old_gray = old_frame[:, :, 0]
    old_gray_cropped = crop_image(old_gray)

    # Detect good features to track
    features_params = dict(maxCorners=5000, qualityLevel=0.3, minDistance=1, blockSize=1)
    p00 = cv2.goodFeaturesToTrack(old_gray_cropped, mask=mask_adjusted_cropped, **features_params)
    p00 = np.vstack((p00, p0_cropped))

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    counter = 0
    for _, filename in enumerate(os.listdir(images_folder)[:]):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(images_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Load images
            new_frame = cv2.imread(image_path)

            image_size = np.array(new_frame.shape)[:-1]
            image_height = image_size.min()

            # Convert to grayscale (using red color channel)
            new_gray = new_frame[:, :, 0]
            new_gray_cropped = crop_image(new_gray)

            # Calculate optical flow
            p1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray_cropped, new_gray_cropped, p00, None, **lk_params)

            # Select good points based on angle and distance deviations
            st = np.ones(st.shape)
            good_new = p1[st == 1]
            good_old = p00[st == 1]

            split = image_height/2

            good_new_lower = good_new[good_new[:,1]<split]
            good_new_upper = good_new[good_new[:,1]>split]
            good_old_lower = good_old[good_new[:,1]<split]
            good_old_upper = good_old[good_new[:,1]>split]
            good_new_list = [good_new_lower, good_new_upper]
            good_old_list = [good_old_lower, good_old_upper]

            good_new_merged = None
            good_old_merged = None

            for good_new, good_old in zip(good_new_list, good_old_list):

                angles = []
                distances = []

                for new, old in zip(good_new, good_old):
                    dx = new[0] - old[0]
                    dy = new[1] - old[1]
                    angle = np.arctan2(dy, dx) * 180 / np.pi
                    if angle < 0:
                        angle += 360
                    angles.append(angle)
                    distances.append(np.sqrt(dx ** 2 + dy ** 2))

                # Calculate the mean and standard deviation of angles and distances
                angles_mean = np.mean(angles)
                angles_std = np.std(angles)
                distances_mean = np.mean(distances)
                distances_std = np.std(distances)

                # Find the indices of outliers based on a threshold of 3 standard deviations
                angle_outliers = np.where(np.abs(angles - angles_mean) > 1 * angles_std)[0]
                distance_outliers = np.where(np.abs(distances - distances_mean) > 1 * distances_std)[0]

                outliers = np.unique(np.concatenate((angle_outliers, distance_outliers)))

                good_new = np.delete(good_new, outliers, axis=0)
                good_old = np.delete(good_old, outliers, axis=0)

                # Uncrop the points
                good_new = uncrop_points(good_new)
                good_old = uncrop_points(good_old)

                if good_new_merged is None:
                    good_old_merged = good_old[np.array(image_height > good_new[:, 1]) & np.array(good_new[:, 1] > 0)]
                    good_new_merged = good_new[np.array(image_height > good_new[:, 1]) & np.array(good_new[:, 1] > 0)]
                else:
                    good_old_merged = np.vstack((good_old_merged, good_old[np.array(image_height > good_new[:, 1]) & np.array(good_new[:, 1] > 0)]))
                    good_new_merged = np.vstack((good_new_merged, good_new[np.array(image_height > good_new[:, 1]) & np.array(good_new[:, 1] > 0)]))

            good_new = good_new_merged
            good_old = good_old_merged

            # Draw the tracks
            for _, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                start_points = (int(a.real), int(b.real))
                end_points = (int(c.real), int(d.real))
                cv2.circle(new_frame, (start_points), 5, (255, 0, 0), -1)
                cv2.line(new_frame, start_points, end_points, (0, 255, 0), 2)
                cv2.circle(new_frame, (end_points), 5, (0, 255, 0), -1)

            # Save the result
            if save_images:
                cv2.imwrite(output_path, new_frame)
            np.save(output_folder_np + filename.replace('jpg', 'npy'), good_new)
            counter += 1
    print(f"Keypoints for {counter} images saved to {output_folder_np}")

