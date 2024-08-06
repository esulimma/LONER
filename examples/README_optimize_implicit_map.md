# Implicit Map Optimization

This document provides a step-by-step guide on how to optimize an implicit map. The main tasks include downloading a Rosbag, adjusting the Rosbag, extracting ground truth (GT) poses, and configuring the optimization algorithm. Follow the instructions carefully to ensure successful data preparation.

## Data Preparation

To get preprocessed_data, to use for the optimization algorithm, two options are available

1. Download a rosbag, a ground truth trajectory .csv and config .yaml file from G-Drive
and place it in the data folder structure
user@docker:~$/data/haveri_hpk/02_02_04/harveri_hpk_02_02_04_adjusted.bag
2. Follow the instructions in data preparation [readme](../rosbag_utils/README_data_preparation.md)


## Optimizer Settings

- CHUNK_SIZE

Represents the amount of rays used for training in parallel, decrease if GPU memory is insufficent

 - ITERATE_LIDAR, LIDAR_MOTION_COMP, MAX_WINDOW_LENGTH_LIDAR

If to train the sigma MLP using lidar data, if to motion compensate the individual lidar scans and the amount of lidar scans used within one optimization window.

 - ITERATION_STRATEGY_LIDAR, NUM_ITERATIONS, L1_THRESHOLD, LIDAR_REPETITIONS

Three different ray sample selecion strategies are available

      'RANDOM' selects a number of n_samples (mapper: optimizer: num_samples: lidar) per scan in the optimization window at random and trains each window for NUM_ITERATIONS.
 
      'MASK' selects a number of n_samples (mapper: optimizer: num_samples: lidar) per scan in the optimization window at random, with a predefined weighing ratio of samples correspoding to the trunks (70 percent) and trains each window for NUM_ITERATIONS.

      'FIXED' selects every available ray per scan in the optimization window for optimization.

The iteration will be repeated until the L1_THRESHOLD or the number of predefined LIDAR_REPETITIONS is reached, or if the evaluation metrics do not improve from the previous iteration.

 - START_STEP_LIDAR, SKIP_STEP_LIDAR, END_STEP_LIDAR

Index reference, start and end index and stepsize for lidar trajectory to be used in optimization.

 - N_EVAL

Number of lidar scans to remove from optimization trajectory, to be used for evaluation of unobserved scans. If lidar scan trajectory should be shuffled before optimization.

 - ITERATE_CAMERA, MAX_WINDOW_LENGTH_CAMERA

If to train the color MLP using camera data and the amount of camera images used within one optimization window.

 - ITERATION_STRATEGY_CAMERA, NUM_ITERATIONS

Two different ray sample selecion strategies are available

      'RANDOM' selects a number of n_samples (mapper: optimizer: num_samples: lidar) per image in the optimization window at random and trains each window for NUM_ITERATIONS.
 
      'FIXED' selects every available ray per image in the optimization window for optimization.

 - IMAGE_UPSAMPLING, IMAGE_MASK, ENLARGE_MASK, MASK_FROM_BAG

Whether or not to upsample the images before optimization, if a mask should be used, if that mask should be enlarged to cover a larger area of the image and if to load dynamic masks, read from a rosbag or if a static mask is to be used.

 - SCHUFFLE

Whether or not to shuffle the trajectories before optimization.


Other options, related to the model configuration, can be adjusted as "changes:" in the .yaml config file, i.e. /haveri_hpk_02_02_04.yaml.
The default .yaml files should be left as is.

      ../cfg/haveri_hpk/haveri_hpk_02_02_04.yaml
      ../cfg/model_config/default_model_config.yaml
      ../cfg/nerf_config/default_nerf_hash.yaml

## Run Implicit Map Optimization Algorithm

Run the algorithm using the following command:

      cd examples
      python3 fdt_optimize_implicit_map.py 
      
      with arguments: path to config .yaml file
            for example: ../cfg/haveri_hpk/haveri_hpk_02_02_04.yaml

      and additional arguments to reiterate and existing map: path to output folder
            for example --experiment_directory ../outputs/02_02_04_baseline_060424_090342

in order to debug the algorithm can be launched using the [launch.json](../.vscode/launch.json) configuration "Optimize Implicit Map"

to segment the map into submaps run:

      fdt_segment_and_optimize_submaps.py 

## Outputs and Analysis
Results are saved as checkpoints in the [outputs folder](../outputs). The .tar files contain the implicit map. In order to evalute the outputs, we need to render scans, images or meshes in order to compare them to the gt data. In order to do this follow the instructions in analysis [readme](../analysis/README_evaluate_results.md)

