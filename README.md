# Forest Digital Twin Preface
## Step-by-Step Instructions
Please refer to the following markdowns for further information on how to preprocess data, optimize implicit maps, visulize results and run a ROS implementation of the implicit maps:

- Data Preprocessing: [readme](/rosbag_utils/README_data_preparation.md)
- Implicit Map Optimization [readme](/examples/README_optimize_implicit_map.md)
- Implicit Map Evaluation [readme](/analysis/README_evaluate_results.md)
- ROS Integration [readme](/gazebo/README_fdt_simulation.md)

## Notes on Docker Layout
The two main folders of the docker layout are 

      user@host:~$/LonerSLAM
and 

      user@host:~$/data

while all imports are designed to be agnostic of the naming of the source and data folder, it is still advised to keep this naming structure.

so when calling from root:

      esulimma@mavt-rsl-n111l:~$ ls

inside the docker it should display

      LonerSLAM  data

# Project Structure

Here is a highlight of the most important scripts and the folder structure

/LonerSLAM\
├── [analysis](/analysis)\
│ ├── [compute_metrics](/analysis/compute_metrics) (contains scripts to compute metrics)\
│ ├── [fdt_analysis](/analysis/fdt_analysis)\
│ │ ├── [fdt_analysis_metrics.py](/analysis/fdt_analysis/fdt_analysis_metrics.py) (script to compute metrics)\
│ │ ├── [fdt_analysis.py](/analysis/fdt_analysis/fdt_analysis.py) (main file for evaluating implicit maps)\
│ ├── [README_evaluate_results.md](/analysis/README_evaluate_results.md)\
├── assets\
├── [cfg](/cfg) (contains config .yaml files)\
├── docker\
├── docs\
├── examples\
│ ├── [fdt_optimize_implicit_map_utils.py](/examples/fdt_optimize_implicit_map_utils.py)\
│ ├── [fdt_optimize_implicit_map.py](/examples/fdt_optimize_implicit_map.py) (main script for optimizing implicit maps)\
│ ├── [README_optimize_implicit_map.md](/examples/README_optimize_implicit_map.md)\
├── gazebo\
│ ├── [fdt_simulation](/gazebo/fdt_simulation) (main script for optimizing implicit maps)\
│ ├── [README_fdt_simulation.md](/gazebo/README_fdt_simulation.md)\
├── [outputs](/outputs) (contains all outputs from implicit map optimization)\
├── [rosbag_utils](/rosbag_utils)  (contains everything necessary for data preprocessing)\
│ ├── [README_data_preparation.md](/rosbag_utils/README_data_preparation.md)\
├── setup_utils\
├── src\
│ ├── [optimizer.py](/src/optimizer.py) (main script for implicit map optimization, including RGB sigma MLP optimization)\
└── [models](/models)\
│ ├── [nerf_tcnn.py](/models/nerf_tcnn.py) (script containing decoupled NeRF model)\
│ ├── [rendering_tcnn.py](/models/rendering_tcnn.py) (script for rendering, including peak rendering)\

/data\
├── haveri_hpk\
│ ├── ... (rosbag location)\
│ ├── ... (rosbag location)\
│ ├── ... (rosbag location)\

# Packages

installed packages outside of the ones listed in the requirements file:

      absl-py==0.14.1
      addict==2.4.0
      alabaster==0.7.12
      ansi2html==1.8.0
      anyio==4.0.0
      apex==0.1
      appdirs==1.4.4
      argcomplete==3.1.2
      argon2-cffi==21.1.0
      arrow==1.3.0
      asgiref==3.4.1
      aspose-3d==24.1.0
      async-lru==2.0.4
      attrs==23.1.0
      audioread==2.1.9
      Babel==2.13.0
      backcall==0.2.0
      backports.functools-lru-cache==1.6.4
      bagpy==0.5
      beautifulsoup4==4.10.0
      bitarray==2.9.2
      bitstring==4.1.4
      bleach==4.1.0
      blis==0.7.4
      brotlipy==0.7.0
      cachetools==4.2.4
      catalogue==2.0.6
      catkin-pkg==1.0.0
      certifi==2021.5.30
      cffi==1.14.6
      chardet==4.0.0
      charset-normalizer==2.0.0
      click==8.0.1
      codecov==2.1.12
      colorama==0.4.4
      comm==0.1.4
      conda==4.10.3
      conda-build==3.21.4
      conda-package-handling==1.7.3
      coverage==6.0.1
      cryptography==3.4.7
      cycler==0.10.0
      cymem==2.0.5
      Cython==0.29.24
      dash==2.13.0
      dash-core-components==2.0.0
      dash-html-components==2.0.0
      dash-table==5.0.0
      dataclasses==0.8
      debugpy==1.5.0
      decorator==5.1.0
      defusedxml==0.7.1
      distro==1.8.0
      Django==3.2.6
      docker-pycreds==0.4.0
      docutils==0.17.1
      empy==4.0.1
      entrypoints==0.3
      evo==1.25.1
      exceptiongroup==1.1.3
      expecttest==0.1.3
      fastjsonschema==2.18.1
      filelock==3.3.0
      flake8==3.7.9
      Flask==2.0.2
      fqdn==1.5.1
      future==0.18.2
      fvcore==0.1.5.post20221221
      genpy==2022.1
      gitdb==4.0.10
      GitPython==3.1.37
      glob2==0.7
      google-auth==1.35.0
      google-auth-oauthlib==0.4.6
      graphsurgeon==0.4.5
      graphviz==0.20.1
      grpcio==1.41.0
      gunicorn==20.1.0
      h11==0.12.0
      httptools==0.2.0
      hypothesis==4.50.8
      idna==3.1
      imagesize==1.2.0
      importlib-metadata==6.8.0
      importlib-resources==6.1.0
      iniconfig==1.1.1
      iopath==0.1.9
      ipykernel==6.4.1
      ipython==7.28.0
      ipython-genutils==0.2.0
      ipywidgets==8.1.1
      isoduration==20.11.0
      itsdangerous==2.0.1
      jedi==0.18.0
      Jinja2==3.0.3
      joblib==1.1.0
      json5==0.9.6
      jsonpointer==2.4
      jsonschema==4.19.1
      jsonschema-specifications==2023.7.1
      jupyter_client==7.4.9
      jupyter_core==5.3.2
      jupyter-events==0.7.0
      jupyter-lsp==2.2.0
      jupyter_server==2.7.3
      jupyter_server_terminals==0.4.4
      jupyter-tensorboard==0.2.0
      jupyterlab==4.0.6
      jupyterlab-pygments==0.1.2
      jupyterlab_server==2.25.0
      jupyterlab-widgets==3.0.9
      jupytext==1.13.0
      kiwisolver==1.3.2
      lark-parser==0.7.8
      lazy_loader==0.3
      libarchive-c==3.1
      librosa==0.8.1
      lightning-utilities==0.9.0
      llvmlite==0.35.0
      lmdb==1.2.1
      lz4==4.3.2
      Markdown==3.3.4
      markdown-it-py==1.1.0
      MarkupSafe==2.1.3
      matplotlib-inline==0.1.3
      mccabe==0.6.1
      mdit-py-plugins==0.2.8
      mistune==3.0.2
      mock==4.0.3
      murmurhash==1.0.5
      natsort==8.4.0
      nbclient==0.5.4
      nbconvert==7.9.2
      nbformat==5.7.0
      nest-asyncio==1.5.8
      networkx==3.1
      nltk==3.6.4
      notebook==6.4.1
      notebook_shim==0.2.3
      numba==0.52.0
      numexpr==2.8.6
      numpy==1.21.2
      nvidia-dali-cuda110==1.6.0
      nvidia-dlprof-pytorch-nvtx==1.6.0
      nvidia-dlprofviewer==1.6.0
      nvidia-pyindex==1.0.9
      oauthlib==3.1.1
      onnx==1.8.204
      overrides==7.4.0
      packaging==23.2
      pandas==2.0.3
      pandocfilters==1.5.0
      parso==0.8.2
      pathtools==0.1.2
      pathy==0.6.0
      pexpect==4.8.0
      pickleshare==0.7.5
      Pillow==10.0.1
      pip==21.2.4
      pkginfo==1.7.1
      pkgutil_resolve_name==1.3.10
      platformdirs==3.11.0
      plotly==5.17.0
      pluggy==1.0.0
      plyfile==1.0.3
      point-cloud-utils==0.30.4
      polygraphy==0.33.0
      pooch==1.5.1
      portalocker==2.3.2
      preshed==3.0.5
      prettytable==2.2.1
      prometheus-client==0.11.0
      prompt-toolkit==3.0.20
      protobuf==3.18.1
      psutil==5.8.0
      ptyprocess==0.7.0
      py==1.10.0
      py3rosmsgs==1.18.2
      pyasn1==0.4.8
      pyasn1-modules==0.2.8
      pybind11==2.8.0
      pycocotools==2.0+nv0.5.1
      pycodestyle==2.11.0
      pycosat==0.6.3
      pycparser==2.20
      pydantic==1.8.2
      pydot==1.4.2
      pyflakes==2.1.1
      Pygments==2.10.0
      pykitti==0.3.1
      pymesh==1.0.2
      pyOpenSSL==20.0.1
      pyparsing==2.4.7
      pyquaternion==0.9.9
      pyrsistent==0.18.0
      pyserial==3.5
      PySocks==1.7.1
      pytest==6.2.5
      pytest-cov==3.0.0
      pytest-pythonpath==0.7.3
      python-dateutil==2.8.2
      python-dotenv==0.19.1
      python-hostlist==1.21
      python-json-logger==2.0.7
      python-nvd3==0.15.0
      python-slugify==5.0.2
      pytools==2023.1.1
      pytorch-quantization==2.1.0
      pytorch3d==0.7.2
      pytz==2021.3
      PyWavelets==1.4.1
      PyYAML==5.4.1
      pyzmq==25.1.1
      referencing==0.30.2
      regex==2021.10.8
      requests==2.31.0
      requests-oauthlib==1.3.0
      resampy==0.2.2
      retrying==1.3.4
      revtok==0.0.3
      rfc3339-validator==0.1.4
      rfc3986-validator==0.1.1
      rosbags==0.9.16
      rpds-py==0.10.4
      rsa==4.7.2
      ruamel.yaml==0.17.35
      ruamel.yaml.clib==0.2.8
      sacremoses==0.0.46
      scikit-image==0.21.0
      scikit-learn==1.0
      scikit-video==1.1.11
      scipy==1.10.1
      seaborn==0.13.0
      Send2Trash==1.8.2
      sentry-sdk==1.31.0
      setproctitle==1.3.3
      setuptools==58.2.0
      setuptools-scm==8.0.4
      shellingham==1.4.0
      six==1.16.0
      smart-open==5.2.1
      smmap==5.0.1
      sniffio==1.3.0
      snowballstemmer==2.1.0
      SoundFile==0.10.3.post1
      soupsieve==2.0.1
      spacy==3.1.3
      spacy-legacy==3.0.8
      Sphinx==4.2.0
      sphinx-glpi-theme==0.3
      sphinx-rtd-theme==1.0.0
      sphinxcontrib-applehelp==1.0.2
      sphinxcontrib-devhelp==1.0.2
      sphinxcontrib-htmlhelp==2.0.0
      sphinxcontrib-jsmath==1.0.1
      sphinxcontrib-qthelp==1.0.3
      sphinxcontrib-serializinghtml==1.1.5
      sqlparse==0.4.2
      srsly==2.4.1
      tabulate==0.8.9
      tenacity==8.2.3
      tensorboard==2.6.0
      tensorboard-data-server==0.6.1
      tensorboard-plugin-wit==1.8.0
      tensorrt==8.0.3.4
      termcolor==2.3.0
      terminado==0.12.1
      testpath==0.5.0
      text-unidecode==1.3
      thinc==8.0.10
      threadpoolctl==3.0.0
      tifffile==2023.7.10
      tinycss2==1.2.1
      tinycudann==1.7
      toml==0.10.2
      tomli==1.2.1
      torch==1.10.0a0+0aef44c
      torch-tb-profiler==0.4.3
      torchmetrics==1.2.0
      torchtext==0.11.0a0
      torchvision==0.11.0a0
      torchviz==0.0.2
      tornado==6.3.3
      tqdm==4.62.3
      traitlets==5.11.2
      trimesh==3.23.5
      typer==0.4.0
      types-python-dateutil==2.8.19.14
      typing_extensions==4.8.0
      tzdata==2023.3
      uff==0.6.9
      uri-template==1.3.0
      urllib3==2.0.6
      uvicorn==0.15.0
      uvloop==0.16.0
      wandb==0.15.12
      wasabi==0.8.2
      watchgod==0.7
      wcwidth==0.2.5
      webcolors==1.13
      webencodings==0.5.1
      websocket-client==1.6.4
      websockets==10.0
      Werkzeug==2.2.3
      wheel==0.37.0
      whitenoise==5.3.0
      widgetsnbextension==4.0.9
      yacs==0.1.8
      zipp==3.17.0
      zstandard==0.21.0


To install the docker image follow the instructions from the original markdown below.

# **LONER**: **L**iDAR **O**nly **Ne**ural **R**epresentations for Real-Time SLAM

### [Paper](https://arxiv.org/abs/2309.04937) | [Project Page](https://bit.ly/loner_slam)

<div align="center">
<img src="./assets/Fig1.png" width="70%" />
</div>

<p align="center">
<strong>Seth Isaacson*</strong>, <strong>Pou-Chun Kung*</strong>, <strong>Mani Ramanagopal</strong>, <strong>Ram Vasudevan</strong>, and <strong>Katherine A. Skinner</strong> <br>
{sethgi, pckung, srmani, ramv, kskin}@umich.edu
</p>

**Abstract**: *This paper proposes LONER, the first real-time LiDAR SLAM algorithm that uses a neural implicit scene representation. Existing implicit mapping methods for LiDAR show promising results in large-scale reconstruction, but either require groundtruth poses or run slower than real-time. In contrast, LONER uses LiDAR data to train an MLP to estimate a dense map in real-time, while simultaneously estimating the trajectory of the sensor. To achieve real-time performance, this paper proposes a novel information-theoretic loss function that accounts for the fact that different regions of the map may be learned to varying degrees throughout online training. The proposed method is evaluated qualitatively and quantitatively on two open-source datasets. This evaluation illustrates that the proposed loss function converges faster and leads to more accurate geometry reconstruction than other loss functions used in depth-supervised neural implicit frameworks. Finally, this paper shows that LONER estimates trajectories competitively with state-of-the-art LiDAR SLAM methods, while also producing dense maps competitive with existing real-time implicit mapping methods that use groundtruth poses.*


## Contact Information

For any questions about running the code, please open a GitHub issue and provide a detailed explanation of the problem including steps to reproduce, operating system details, and hardware. Please open issues with feature requests, we're happy to help you fit the code to your needs!

For research inquiries, please contact one of the lead authors:

- Seth Isaacson: sethgi [at] umich [dot] edu
- Pou-Chun (Frank) Kung: pckung [at] umich [dot] edu


## Running the Code

### Prerequisites
This has been tested on an Ubuntu 20.04 docker container. We highly recommend you use our docker configuration. If you have specific needs for running outside of docker, please open an issue and we'll work on documentation for how to do that. You will need:

1. Docker: https://docs.docker.com/engine/install/
2. nvidia-docker2: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### Using the Docker Container
This repository has everything you need to run with docker. 


#### Building the container

```
cd <project_root>/docker
./build.sh
```

This will pull an [image](https://hub.docker.com/layers/sethgi/loner/base_1.0/images/sha256-b86796e44ccac26bcaa914a20bd16bf2068bfbb8278761b8f51d71e16fcdd6f5?context=repo) from DockerHub, then make local modifications for the user.


#### Data Preparation
By default, we assume all data has been placed in `~/Documents/LonerSlamData`. If you have data in a different, you can go into `docker/run.sh` and change `DATA_DIR` to whatever you want. If you need multiple directories mounted, you'll need to modify the run script.


#### Run Container

To run the container, `cd docker` and `./run.sh`. The `run.sh` file has the following behavior:

- If no container is currently running, a new one will be started.
- If a container is already running, you will be attached to that. Hence, running `./run.sh` from two terminals will connect you to a single docker container.
- If you run with `./run.sh restart`, the existing container (if it exists) will be killed and removed and a new one will be started.

#### VSCode
This repo contains everything you need to use the Docker extension in VSCode. To get that to run properly:
1. Install the docker extension.
2. Reload the workspace. You will likely be prompted if you want to re-open the folder in a dev-container. Say yes.
3. If not, Click the little green box in the bottom left of the screen and select "Re-open Folder in Dev Container"
4. To make python recognize everything properly, go to the python environment extension (python logo in the left toolbar) and change the environment to Conda Base 3.8.12.

The DevContainer provided with this package assumes that datasets are stored in `~/Documents/LonerSlamData`. If you put the data somewhere else, modify the `devcontainer.json` file to point to the correct location.

When you launch the VSCode DevContainer, you might need to point VSCode manually to the workspace when prompted. It's in `/home/$USER/LonerSLAM`


## Running experiments

### Download Fusion Portable
Download the sequences you care about from [fusion portable](https://fusionportable.github.io/dataset/fusionportable), along with the 20220209 calibration.
We have tested on 20220216_canteen_day, 20220216_garden_day, and 20220219_MCR_normal_01.

Put them in the folder you pointed the docker scripts (or VSCode `devcontainer.json` file) to mount (by default `~/Documents/LonerSlamData`). Also, download the groundtruth data.

Now, modify `cfg/fusion_portable/<sequence_name>.yaml` to point to the location of the data, as seen from within the Docker container. So, if you clone the data to `~/Documents/LonerSlamData/<sequence>`, docker will mount that to `~/data/<sequence>`.

Finally, `cd examples` and `python3 run_loner.py ../../cfg/<sequence_name>.yaml`.

The results will be stored into `outputs/<experiment_name>_<timestamp>/` where `<experiment_name>` is set in the configuration file, and `<timestamp>=YYMMDD_hhmmss`

### Run

Run the canteen sequence in Fusion Portable:

```
cd examples
python3 run_loner.py ../cfg/fusion_portable/canteen.yaml
```

There are several ways to visualize the results. Each will add new files to the output folder.

#### Visualize Map
Render depth images:
```
python3 renderer.py ../outputs/<output_folder> 
```
The `renderer.py` file is also capable of producing videos by adding the `--render_video` flag. Run `renderer.py --help` for a full description of the options. 

#### Meshing the scene:
```
python3 meshing.py ../outputs/<output_folder> ../cfg/fusion_portable/canteen.yaml \
 --resolution 0.1 --skip_step 3 --level 0.1 --viz --save
```

#### Render a lidar point cloud:
This will render a LiDAR point cloud from the frame of the last. This will render LiDAR point clouds from every Nth KeyFrame, then assemble them. N defaults to 5, but can be set with --skip_step N.
```
python3 renderer_lidar.py ../outputs/<output_folder> --voxel_size 0.1
```

Results will be stored in each output folder.

#### Visualize Trajectory
##### 2D visualization using matplotlib
```
python3 plot_poses.py ../outputs/<output_folder>
```

A plot will be stored in `poses.png` in the output folder.

##### 3D visualization using evo

Download the groundtruth trajectories from the dataset in TUM format. Fusion Portable provides those [here](http://filebrowser.ram-lab.com/share/S-2Th4iV). Put them in `<sequence_folder>/ground_truth_traj.txt`:

```
mv <path_to_groundtruth>/traj/20220216_canteen_day.txt \
      ~/data/fusion_portable/20220216_canteen_day/ground_truth_traj.txt 
```

Then prepare the output files:
```
mkdir results && cd results
python3 ~/LonerSLAM/analysis/compute_metrics/traj/prepare_results.py \
      ~/LonerSLAM/outputs/<output_folder>\
      eval_traj canteen \
       ~/data/fusion_portable/20220216_canteen_day/ground_truth_traj.txt  \
       --single_trial --single_config
```

Run trajectory evaluation and visualization:
```
cd results
evo_traj tum ./eval_traj/canteen/stamped_traj_estimate0.txt \
      --ref ./eval_traj/canteen/stamped_groundtruth.txt \
      -a --t_max_diff 0.1 -p
```

This is the barebones way to produce plots, but there are lots of options for qualitative and quantitative comparisons. See the [metrics readme](analysis/compute_metrics/README.md) for more details on computing metrics.

#### Analyzing the Results
Dense trajectories are stored to `<output_dir>/trajectory`. This will contain three files:

1. `estimated_trajectory.txt`: This is the dense trajectory as estimated by LonerSLAM. 
2. `keyframe_trajectory.txt`: This includes only the keyframe poses.
3. `tracking_only.txt`: This is the result of accumulating tracking before any optimization occurs.

To compute metrics, see the information in the [metrics readme](analysis/compute_metrics/README.md) for more details on computing metrics.

## BibTeX

This work has been accepted for publication in the IEEE Robotics and Automation Letters. Please cite as follows:

```
@ARTICLE{loner2023,
 author={Isaacson, Seth and Kung, Pou-Chun and Ramanagopal, Mani and Vasudevan, Ram and Skinner, Katherine A.},
 journal={IEEE Robotics and Automation Letters},
 title={LONER: LiDAR Only Neural Representations for Real-Time SLAM},
 year={2023},
 volume={8},
 number={12},
 pages={8042-8049},
 doi={10.1109/LRA.2023.3324521}}
```

## License


<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="http://github.com/umautobots/loner">LONER</a> by Ford Center for Autonomous Vehicles at the University of Michigan is licensed under <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-SA 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1"><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/sa.svg?ref=chooser-v1"></a></p>

For inquiries about commercial licensing, please reach out to the authors.
