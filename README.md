# Deep Multi-Modal Cooperative 3D Object Detection of Traffic Participants Using Onboard and Roadside Sensors

## Abstract

It is necessary to ensure that the deployed autonomous driving systems are as robust, reliable, and safe as possible. Most research today is focused on onboard vehicle autonomous driving. This approach inherently is weak against occlusion. Cooperative driving, combining vehicle and infrastructure sensors, alleviates this weakness. Camera-LiDAR fusion is used to cover the weaknesses of each sensor modality. However, this increases the amount of data that needs to be processed. As such, it is necessary to find a fusion method that provides the best trade-off between accuracy and speed. Motivated by this, this work proposes a deep, cooperative, camera-LiDAR fusion method in a unified bird’s eye view perspective for 3D object detection. A new TraffiX Cooperative Dataset was also created. This proposed deep cooperative multi-modal method provides the best performance over any other combination that doesn’t use all cameras and LiDARs on both the vehicle and infrastructure for the TraffiX Cooperative Dataset. The proposed method is also proven to be better in infrastructure-only, camera-LiDAR fusion mode than the InfraDet3D method for the TUMTraf Intersection Dataset. Finally, this work provides recommendations to improve the proposed method further in the future.

## Usage

### Prerequisites

The code is built with following libraries:

- Python >= 3.8, \<3.9
- OpenMPI = 4.0.4 and mpi4py = 3.0.3 (Needed for torchpack)
- Pillow = 8.4.0 (see [here](https://github.com/mit-han-lab/bevfusion/issues/63))
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.9, \<= 1.10.2
- [tqdm](https://github.com/tqdm/tqdm)
- [torchpack](https://github.com/mit-han-lab/torchpack)
- [mmcv](https://github.com/open-mmlab/mmcv) = 1.4.0
- [mmdetection](http://github.com/open-mmlab/mmdetection) = 2.20.0
- [nuscenes-dev-kit](https://github.com/nutonomy/nuscenes-devkit)
- Latest versions of numba, torchsparse, pypcd, and Open3D

After installing these dependencies, please run this command to install the codebase:

```bash
python setup.py develop
```

The easiest way to deal with prerequisites is to use the [Dockerfile](docker/Dockerfile). Please make sure that `nvidia-docker` is installed on your machine. After that, please execute the following command to build the docker image:

```bash
cd docker && docker build . -t coopdet3d
```

The docker can then be run with the following command:

```bash
nvidia-docker run -it  -v `pwd`/../data/traffix_coop:/home/coopdet3d/data/traffix_coop -v <PATH_TO_COOPDET3D>:/home/coopdet3d --shm-size 16g coopdet3d /bin/bash
```

It is recommended for users to run data preparation (instructions are available in the next section) outside the docker if possible. Note that the dataset directory should be an absolute path. Within the docker, please run the following command to clone the repo and install custom CUDA extensions:

```bash
cd /home/coopdet3d
python setup.py develop
```

You can then create a symbolic link `data/tumtraf_i` to `/dataset/tumtraf_i` and `data/traffix_coop` to `/dataset/traffix_coop` in the docker.

### Data Preparation

#### TUMTraf Intersection Dataset

Run this script for data preparation:

```bash
python ./tools/create_traffix_data.py --root-path /home/coopdet3d/data/tumtraf_i --out-dir /home/coopdet3d/data/tumtraf_i_processed
```

After data preparation, you will be able to see the following directory structure (as is indicated in mmdetection3d):

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── tumtraf_i
|   |   ├── test
|   |   ├── train
|   |   ├── val
|   ├── tumtraf_i_processed
│   │   ├── traffix_nusc_gt_database
|   |   ├── test
|   |   ├── train
|   |   ├── val
│   │   ├── traffix_nusc_infos_train.pkl
│   │   ├── traffix_nusc_infos_val.pkl
│   │   ├── traffix_nusc_infos_test.pkl
│   │   ├── traffix_nusc_dbinfos_train.pkl

```

#### TraffiX Cooperative Dataset

Before running, make sure that in ./tools/data_converter/create_traffix_gt_database.py lines 155-157 and 173-175 the correct line is uncommented for the dataset type.

Run this script for data preparation:

```bash
python ./tools/create_traffixcoop_data.py --root-path /home/coopdet3d/data/traffix_coop --out-dir /home/coopdet3d/data/traffix_coop_processed --splits testing
```

After data preparation, you will be able to see the following directory structure (as is indicated in mmdetection3d):

```
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── traffix_coop
|   |   ├── test
|   |   ├── train
|   |   ├── val
|   ├── traffix_coop_processed
│   │   ├── traffix_nusc_coop_gt_database
|   |   ├── test
|   |   ├── train
|   |   ├── val
│   │   ├── traffix_nusc_coop_infos_train.pkl
│   │   ├── traffix_nusc_coop_infos_val.pkl
│   │   ├── traffix_nusc_coop_infos_test.pkl
│   │   ├── traffix_nusc_coop_dbinfos_train.pkl

```

### Evaluation

For evaluation on the TUMTraf Intersection Dataset, run:

```bash
torchpack dist-run -np 1 python tools/test.py <PATH_TO_CONFIG_FILE> <PATH_TO_PTH_FILE> --eval bbox
```

For evaluation on the TraffiX Cooperative Dataset, run:

```bash
torchpack dist-run -np 1 python tools/test_coop.py <PATH_TO_CONFIG_FILE> <PATH_TO_PTH_FILE> --eval bbox
```

### Training

NOTE: If you want to use a YOLOv8 .pth file from MMYOLO, please make sure the keys inside fit with this model's structure. Use the file ./tools/convert_yolo_checkpoint.py on that .pth file first. The paths are currently hardcoded in the file, so change it there accordingly.

For training camera-only model on the TUMTraf Intersection Dataset, run:

```bash
torchpack dist-run -np 3 python tools/train.py <PATH_TO_CONFIG_FILE> --model.encoders.camera.backbone.init_cfg.checkpoint  <PATH_TO_PRETRAINED_CAMERA_PTH> 
```

For training LiDAR-only model on the TUMTraf Intersection Dataset, run:

```bash
torchpack dist-run -np 3 python tools/train.py <PATH_TO_CONFIG_FILE>
```

For training fusion model on the TUMTraf Intersection Dataset, run:

```bash
torchpack dist-run -np 3 python tools/train.py <PATH_TO_CONFIG_FILE> --model.encoders.camera.backbone.init_cfg.checkpoint <PATH_TO_PRETRAINED_CAMERA_PTH> --load_from <PATH_TO_PRETRAINED_LIDAR_PTH>
```

For training camera-only model on the TraffiX Cooperative Dataset, run:

```bash
torchpack dist-run -np 3 python tools/train_coop.py <PATH_TO_CONFIG_FILE> --model.vehicle.fusion_model.encoders.camera.backbone.init_cfg.checkpoint  <PATH_TO_PRETRAINED_CAMERA_PTH> --model.infrastructure.fusion_model.encoders.camera.backbone.init_cfg.checkpoint  <PATH_TO_PRETRAINED_CAMERA_PTH> 
```

Use the pretrained camera parameters depending on which type of model you want to train: vehicle-only, camera-only, or cooperative (both). For YOLOv8 camera backboned models, these two flags are optional since the correct weights are already hardcoded to the config files.

For training LiDAR-only model on the TraffiX Cooperative Dataset, run:

```bash
torchpack dist-run -np 3 python tools/train_coop.py <PATH_TO_CONFIG_FILE>
```

For training fusion model on the TraffiX Cooperative Dataset, run:

```bash
torchpack dist-run -np 3 python tools/train_coop.py <PATH_TO_CONFIG_FILE> ---model.vehicle.fusion_model.encoders.camera.backbone.init_cfg.checkpoint  <PATH_TO_PRETRAINED_CAMERA_PTH> --model.infrastructure.fusion_model.encoders.camera.backbone.init_cfg.checkpoint  <PATH_TO_PRETRAINED_CAMERA_PTH> --load_from <PATH_TO_PRETRAINED_LIDAR_PTH>
```
Use the pretrained camera parameters depending on which type of model you want to train: vehicle-only, camera-only, or cooperative (both). For YOLOv8 camera backboned models, these two flags are optional since the correct weights are already hardcoded to the config files.

Note: please run `tools/test.py` separately after training to get the final evaluation metrics.

### Running CoopDet3D inference on TraffiX and save detections in OpenLABEL format 

To run inference of this model, run this command:

```bash
torchpack dist-run -np 1 python scripts/cooperative_multimodal_3d_detection.py <PATH_TO_CONFIG_FILE> --checkpoint <PATH_TO_CHECKPOINT_PTH> --split [train, val, test] --input_type hard_drive --save_detections_openlabel --output_folder_path_detections <PATH_TO_OPENLABEL_OUTPUT_FOLDER>
```
Example:
```
torchpack dist-run -np 1 python scripts/cooperative_multimodal_3d_detection.py configs/traffix_coop/det/transfusion/secfpn/cooperative/camera+lidar/yolov8/pointpillars.yaml  --checkpoint weights/bevfusion_coop_vi_cl_pointpillars512_2x_yolos.pth --split test --input_type hard_drive --save_detections_openlabel --outpu
t_folder_path_detections inference
```

### Built in visualization:

For TUMTraf Intersection Dataset:

```bash
torchpack dist-run -np 1 python tools/visualize.py <PATH_TO_CONFIG_FILE> --checkpoint <PATH_TO_PTH_FILE> --split test --mode pred --out-dir viz_traffix 
```

For TraffiX Cooperative Dataset:

```bash
torchpack dist-run -np 1 python tools/visualize_coop.py <PATH_TO_CONFIG_FILE> --checkpoint <PATH_TO_PTH_FILE> --split test --mode pred --out-dir viz_traffix 
```

For split, naturally one could also choose "train" or "val". For mode, the other options are "gt" (ground truth) or "combo" (prediction and ground truth).

### Benchmarking:

For TUMTraf Intersection Dataset:

```bash
torchpack dist-run -np 1 python tools/benchmark.py <PATH_TO_CONFIG_FILE> <PATH_TO_PTH_FILE> --log-interval 50
```

For TraffiX Cooperative Dataset:

```bash
torchpack dist-run -np 1 python tools/benchmark_coop.py <PATH_TO_CONFIG_FILE> <PATH_TO_PTH_FILE> --log-interval 10
```

### Export to OpenLABEL:

For TUMTraf Intersection Dataset:

```bash
torchpack dist-run -np 1 python tools/inference_to_openlabel.py <PATH_TO_CONFIG_FILE> --checkpoint <PATH_TO_PTH_FILE> --split test --out-dir /home/coopdet3d/openlabel_out/

```

For TraffiX Cooperative Dataset:

```bash
torchpack dist-run -np 1 python tools/inference_to_openlabel_coop.py <PATH_TO_CONFIG_FILE> --checkpoint <PATH_TO_PTH_FILE> --split test --out-dir /home/coopdet3d/openlabel_out/
```

For split, naturally one could also choose "train" or "val".
