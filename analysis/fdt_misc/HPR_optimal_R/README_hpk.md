# Data Preparation

This document provides a step-by-step guide to prepare data for your project. The main tasks include downloading a Rosbag, adjusting the Rosbag, extracting ground truth (GT) poses, and configuring the optimization algorithm. Follow the instructions carefully to ensure successful data preparation.

## Download Rosbag

Download the required Rosbag files using a tool such as wget or directly from the source.
Haveri HPK rosbags can be found under [raw_data](https://drive.google.com/drive/u/0/folders/1HrJFEgE_Za-HeBcwKJwrDM-xhsb2fpNw) folder in Google Drive:

      MA - Erik - Forest Digital Twin/06_Evaluation/raw_data

After downloading the desired Rosbag, place it in the data folder in this manner:

      user@docker:~$/data/haveri_hpk/02_02_04/harveri_hpk_02_02_04.bag

## Adjust Rosbag
In order to run the optimization algorithm for haveri-hpk rosbags, we need to make two major adjustments:
1. Removing points, corresponding to the harvester, from lidar scans.
2. Estimating masks, corresponding to the harvester's articulated arm, from images.


### Remove Points from Lidar Scans
The function: 

      read_and_adjust_bag(input_bag_file, box_dimensions, box_origin, lidar_topic)

reads lidar scan messages and calls: 

      remove_box_from_point_cloud(point_cloud, box_dimensions, box_origin)

to remove points corresponding to a bounding box, as visualized with the removed points in grey:
<div align="center">
<img src="./../assets/image_scan_adjusted.png" width="40%" />
</div>

The adjusted point clouds are written as a lidar scan message to a copy of the rosbag, named input_bag_adjusted, along with all other topics contained in the input_bag
      
      defaults:   - lidar_topic = '/hesai/pandar'
                  - box_dimensions = (4.5, 7, 60)
                  - box_origin = (0, 0.5, 2.5)

### Estimate and add Image Mask

First extract images corresponding to the desired image rostopic using:

      extract_images.save_compressed_images(bag_file_path, topic, output_dir)

These images will be saved to 
      user@docker:~$/data/haveri_hpk/02_02_04/images/ 
      
Afterwards Lucas-Kanade [optical image flow keypoint tracking algorithm](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html) is called to create dynamic input points, tracking the movement of the articulated arm while driving, for the mask estimation:

      image_detect_keypoints.lucas_kanade(images_folder, reference_mask, save_images)

<div align="center">
<img src="./../assets/image_tracked.jpg" width="50%" />
</div>

The tracked images are saved as .jpg files to /data/haveri_hpk/02_02_04/images_tracked/ (if save_images = True) and the keypoints are saved as numpy arrays to /data/haveri_hpk/02_02_04/image_keypoints/

A [reference mask](https://www.remove.bg/upload), corresponding to the first image, drastically improves performance, as it ensures that the keypoints to be tracked are all placed on the articulated arm:

<div align="center">
<img src="./../assets/00_reference_mask.png" width="50%" />
</div>

The ViT-H image encoding algorithm Segment Anything [(SAM)](https://segment-anything.com/) is applied in order to estimate image masks corresponding to the articulated arm

      image_segmentation.vit_h(images_folder, reference_image_filename, dynamic_keypoints, save_images)

This requires sam_model_registry sam_vit_h_4b8939.pth placed in /LonerSLAM/rosbag_utils/

      wget -q \ 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'

A reference image, with a high-contrast background, improves performance, if available in a first step a mask will be estimated for the reference image, which will then be used as an input for the following image:

<div align="center">
<img src="./../assets/00_reference_image.png" width="50%" />
</div>

The segmented images are saved as .jpg files to /data/haveri_hpk/02_02_04/images_segmented/ (if save_images = True) and as numpy arrays to /data/haveri_hpk/02_02_04/image_masks/

<div align="center">
<img src="./../assets/image_segmented.jpg" width="50%" />
</div>

The numpy arrays of the image mask are converted to compressed image ros messages and written to the rosbag containing the images with a given mask topic using:

      read_and_insert_masks(input_bag_file, image_topic, mask_topic)

Using uint8 encoding and the header.stamp of the image message

      defaults:   - topic = "/zed2i/zed_node/rgb/image_rect_color/compressed"
                  - mask_topic = "/zed2i/zed_node/rgb/image_rect_color/mask"
                  - reference_mask = "00_mask.png"
                  - reference_image_filename = "00_reference_image.png"

## Extract Calibration
In order to get the intrinsic and extrinsic camera calibration, call the function:

      extract_calibration.get_calibration(bag_file)

which prints the calibration in the terminal with the following format:

      calibration:
            lidar_to_camera:
                  xyz: [x, y, z]
                  orientation: [qx, qy, qz, qw]
            camera_intrinsic:
                  k: [[fx, 0.0, cx],[0.0, fy, cy],[ 0.0,  0.0, 1.0]]
                  distortion: [k1, k2, t1, t2, k3]
                  new_k: NULL
                  width: w
                  height: h

## Get Ground Truth Poses
In order to run the optimization algorithm, a .csv or .txt file containing GT poses in TUM format 
needs to be provided
Additionally, if a map comparison is desired, a GT map needs to be supplied.
Since the haveri-hpk rosbags may not contain GT poses, they be can estimated them using ICP
This can for example be achieved using [Open3DSLAM](https://github.com/leggedrobotics/open3d_slam)

In the launch file add

      <arg name="cloud_topic" default="/hesai/pandar"/>
      <arg name="bag_filename" default="harveri_hpk_02_02_04.bag"/>

And in the param file set

      params.saving.save_map = true

In order to record the estimated odometry launch

      rosbag record /mapping_node/scan2map_transform /mapping_node/scan2scan_transform 
      /mapping_node/scan2map_odometry /tf

      roslaunch open3d_slam_ros launch_file.launch

The map will be saved in /opend3d_slam/src/ros/open3d_slam_ros/data/maps as map.pcd, rename it to map_GT.pcd and place it in /data/haveri_hpk/02_02_04/ alongside with the rosbag containing the ground truth trajectory harveri_hpk_02_02_04_GT.bag, the following function can then be called to extract the trajectory in TUM format to a .csv file:

      extract_trajectories.extract_gt_o3d_tum(bag_file, output_file, tf_topic, remove_offset)

      defaults:   - tf_topic = "/mapping_node/scan2map_transform"
                  - remove_offset = False

## One-Time Function Call

Assuming the rosbag containing image and lidar data, as well as the rosbag containing the ground truth trajectory are available, all of the steps described above

      1. Remove Points from Lidar Scan
      2. Estimate and add Image Mask
      3. Extract Calibration
      4. Extract GT Poses

can be achieved with a single function call:
      
      adjust_bags.adjust_and_extract_all(input_bag_path, input_gt_bag_path)


## Adjust Configuration File

A few adjustments may need to be done in the config.yaml file placed in /LonerSLAM/cfg/haveri_hpk/02_02_04_baseline.yaml
make sure the following are correct

      dataset: ~/data/haveri_hpk/02_02_04/harveri_hpk_02_02_04_adjusted.bag
      groundtruth_traj: ~/data/haveri_hpk/02_02_04/ground_truth_traj.csv
      experiment_name: ...
      changes:
            calibration: ...
            ros_names: ...

Ensure the paths and parameters are correctly set for your specific setup, including paths to the adjusted Rosbag and GT poses, and any other configuration parameters needed

