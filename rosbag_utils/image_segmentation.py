import os
import sys
import tqdm
import glob

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

import numpy as np
import cv2
import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

import matplotlib.pyplot as plt
import textwrap

# wget -q \
# 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    os.pardir))
sys.path.append(PROJECT_ROOT)

data_path = os.path.dirname(PROJECT_ROOT) + '/data'
ckpt_path = PROJECT_ROOT + '/rosbag_utils/' + "sam_vit_h_4b8939.pth"

SCORE_THRESHOLD = 0.9
SHOW_FIGURES = False

def show_mask(mask, ax, random_color=False):
    """
    Display a binary mask on a given matplotlib axis.

    Parameters:
    - mask: numpy.ndarray
        The binary mask to be displayed.
    - ax: matplotlib.axes.Axes
        The matplotlib axis on which to display the mask.
    - random_color: bool, optional
        If True, a random color will be used for the mask. Default is False.

    Returns:
    None
    """
    # Check if random_color is True
    if random_color:
        # Generate a random color with an alpha value of 0.6
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        # Use a predefined color (Dodger Blue) with an alpha value of 0.6
        color = np.array([30/255, 144/255, 255/255, 0.6])

    # Get the height and width of the mask
    h, w = mask.shape[-2:]

    # Reshape the mask and color arrays to match the dimensions of the mask
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    # Display the mask on the given axis
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=275):
    """
    Display the points on a scatter plot with different colors based on their labels.

    Parameters:
    - coords (numpy.ndarray): The coordinates of the points.
    - labels (numpy.ndarray): The labels of the points.
    - ax (matplotlib.axes.Axes): The axes object to plot on.
    - marker_size (int, optional): The size of the markers. Default is 275.

    Returns:
    None
    """

    # Separate positive and negative points based on their labels
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]

    # Plot positive points in green with a star marker
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    # Plot negative points in red with a star marker
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    
def show_box(box, ax):
    """
    Display a bounding box on a matplotlib Axes object.

    Parameters:
    - box (list): A list containing the coordinates of the bounding box in the format [x0, y0, x1, y1].
    - ax (matplotlib.axes.Axes): The Axes object on which to display the bounding box.

    Returns:
    None
    """

    # Extract the coordinates of the bounding box
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]

    # Add a rectangle patch to the Axes object
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


def vit_h(images_folder, reference_image_filename='00_reference_image.png', 
          dynamic_keypoints=True, save_images=False):
    """
    Perform image segmentation using the ViT-H model. Segmented images and masks saved in the output folder.

    Args:
        images_folder (str): Path to the folder containing the input images.
        reference_image_filename (str, optional): Name of the reference image file. Defaults to '00_reference_image.png'.
        dynamic_keypoints (bool, optional): Flag indicating whether to use dynamic keypoints. Defaults to True.
        save_images (bool, optional): Flag indicating whether to save the segmented images. Defaults to False.
    Returns:
        None
    """

    # Create output folders for segmented images and masks
    output_folder = images_folder.replace('images', 'images_segmented') + "/"
    os.makedirs(output_folder, exist_ok=True)
    output_folder_masks = images_folder.replace('images', 'image_masks') + "/"
    os.makedirs(output_folder_masks, exist_ok=True)

    # Set the model type and device
    model_type = "vit_h"
    device = "cuda"
    torch.cuda.init()
    print("---------------------------------------------------------------------")
    print(f"Segmenting Images from {images_folder}")

    try:
        # Load the segmentation model
        sam = sam_model_registry[model_type](checkpoint=ckpt_path)
        sam.to(device=device)
        sam.to(0)
    except:
        raise ImportError(textwrap.fill(textwrap.dedent(f"""
        Unable to load model {model_type} 
        From path {ckpt_path}
        Try:
        wget -q \ 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'
        """)))

    # Get the path of the reference image
    reference_image_path = os.path.join(os.path.dirname(images_folder), reference_image_filename)

    if not os.path.exists(reference_image_path):
        reference_image_path = os.path.join(os.path.dirname(os.path.dirname(images_folder)), reference_image_filename)

    if os.path.exists(reference_image_path):
        print(f"Reference image found at {reference_image_path}")
    else:
        print(f"Reference image not found at {reference_image_path}")
        print(f"Creating reference image from first image in folder {images_folder}")
        reference_image_path = os.path.join(images_folder, os.listdir(images_folder)[0])

    image = cv2.imread(reference_image_path)

    predictor = SamPredictor(sam)
    predictor.set_image(image)

    # Define input points and labels for segmentation
    input_point = np.array([[215, 320], [200, 165], [190, 50]])
    input_label = np.array([1, 1, 1])

    # Predict Mask for Image
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True)

    # Assign the mask with the highest score as the input mask
    mask_input = logits[np.argmax(scores), :, :]

    # Define input points and labels for segmentation
    input_point = np.array([[215, 320], [200, 290], [200, 165], [190, 100], [190, 50]])
    input_label = np.array([1, 1, 1, 1, 1])

    # Define input box for segmentation
    input_box = np.array([92, 360, 360, 0])

    # Repredict Mask for Image
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        mask_input=mask_input[None, :, :],
        multimask_output=True,
    )

    # Assign the mask with the highest score as the input mask
    mask_input = logits[np.argmax(scores), :, :]
    mask = masks[np.argmax(scores), :, :]

    # Show figures if enabled
    if SHOW_FIGURES:
        plt.figure()
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        show_box(input_box, plt.gca())
        plt.title(f"Mask , Score: {np.max(scores):.3f}", fontsize=18)
        plt.axis('off')
        plt.show

    acc_mask = np.ones((360, 640), dtype=bool)

    file_list = list(filter(os.path.isfile, glob.glob(images_folder + "*.jpg")))
    img_format = 'jpg'

    if file_list == []:
        file_list = list(filter(os.path.isfile, glob.glob(images_folder + "*.png")))
        img_format = 'png'

    file_list.sort(key=lambda x: os.path.getmtime(x))

    WARN_ONCE_KEYPOINTS = False
    WARN_ONCE_SCORE = False
    below_threshold_counter = 0
    mask_counter = 0

    input_point_static = input_point
    input_label_static = input_label

    for filename in tqdm.tqdm(file_list, desc="Segmenting Images"):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(images_folder, filename)
            image = cv2.imread(image_path)

            if dynamic_keypoints:
                keypoint_path = filename.replace('images', 'image_keypoints').replace(img_format, 'npy')
                if os.path.exists(keypoint_path):
                    # Define input points and labels for segmentation    
                    input_point_dyn = np.load(keypoint_path)
                    input_label_dyn = np.ones((input_point_dyn.shape[0],))
                    input_point = np.concatenate([input_point_static, input_point_dyn], axis=0)
                    input_label = np.concatenate([input_label_static, input_label_dyn], axis=0)
                else:
                    if not WARN_ONCE_KEYPOINTS:
                        WARN_ONCE_KEYPOINTS = True
                        print(f"No Dynamic Keypoints found for: {filename}")
                        print("Using Static Keypoints instead.")
                        print("Run image_detect_keypoints.lucas_canade to generate Dynamic Keypoints")
                        print("For Improved Performance")
                        print("Will not show this warning again")
                    input_point = input_point_static
                    input_label = input_label_static

            # Predict Mask for Image
            predictor.set_image(image)
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                mask_input=mask_input[None, :, :],
                multimask_output=True,
            )

            # Check if the maximum score is below the threshold
            if scores.max() < SCORE_THRESHOLD:
                if not WARN_ONCE_SCORE:
                    WARN_ONCE_SCORE = True
                    print(f"Mask Score for {filename}:")
                    print(f"{np.max(scores):.3f}, below Threshold: {SCORE_THRESHOLD}")
                    print("Assigning Previous Mask")
                    print("Will not show this warning again")
                below_threshold_counter += 1
            else:
                # Assign the mask with the highest score as the input mask
                mask = masks[np.argmax(scores), :, :]
                mask_input = logits[np.argmax(scores), :, :]

            # Show figures if enabled
            if SHOW_FIGURES:
                plt.figure(figsize=(10, 10))
                plt.figure()
                plt.imshow(image)
                show_mask(mask, plt.gca())
                show_points(input_point, input_label, plt.gca())
                show_box(input_box, plt.gca())
                plt.title(f"Mask , Score: {np.max(scores):.3f}", fontsize=18)
                plt.axis('off')
                plt.show()

            segmented_image = cv2.imread(image_path)

            # Apply the mask to the image (red channel)
            segmented_image[:, :, 2][mask] = segmented_image[:, :, 0][mask] + segmented_image[:, :, 1][mask] + segmented_image[:, :, 1][mask]
            segmented_image[:, :, 0][mask] = segmented_image[:, :, 0][mask] * 0.01
            segmented_image[:, :, 1][mask] = segmented_image[:, :, 1][mask] * 0.01
            segmented_image[:, :, 2][mask] = segmented_image[:, :, 2][mask] * 0.9

            # Draw circles around the input points
            for point in input_point:
                cv2.circle(segmented_image, (int(point[0]), int(point[1])), 3, (0, 0, 0), 2)
                cv2.circle(segmented_image, (int(point[0]), int(point[1])), 5, (0, 0, 215), 2)

            # Add text to the image, representing the mask score
            image_size = np.array(segmented_image.shape)[:-1]
            image_height = image_size.min()
            image_width = image_size.max()
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (0, 0, 255)
            thickness = 1
            text = f'Mask Score: {np.max(scores):.3f}'
            text_size, _ = cv2.getTextSize(text, font, fontScale, thickness)
            text_w, text_h = text_size
            org = (image_width - text_w, image_height - text_h)

            cv2.rectangle(segmented_image, (org[0], org[1] + int(text_h / 2) + 6),
                          (org[0] + text_w, org[1] - text_h + 8), (0, 0, 0), -1)
            cv2.putText(segmented_image, text, (org[0], org[1] + int(text_h / 2)), font, fontScale, color, thickness,
                        cv2.LINE_AA)

            # Save the segmented image
            if save_images:
                cv2.imwrite(filename.replace('images', 'images_segmented').replace('image_', 'mask_'), segmented_image)

            # Invert the mask and save it as a numpy file
            inverted_mask = np.logical_not(mask)
            np.save(filename.replace('images', 'image_masks').replace(img_format, 'npy'), inverted_mask)

            # add mask to accumulated mask
            acc_mask = np.logical_and(acc_mask, inverted_mask)
            mask_counter += 1

    print(f"Number of images segmented: {mask_counter}")
    print(f"Number of masks with score below threshold: {below_threshold_counter}")
    
    # Save the accumulated mask
    if save_images:
        cv2.imwrite(os.path.dirname(os.path.dirname(images_folder)) +  "/accumulated_mask.png", acc_mask.astype(np.uint8) * 255)
    np.save(os.path.dirname(os.path.dirname(images_folder)) + "/accumulated_mask.npy", acc_mask)
    print("Accumulated mask saved")