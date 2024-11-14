# VLDUS: Vision-Language Distillated Unseen Synthesizer for Zero-Shot Object Detection

## 📢 Description

This is the code repository related to "VLDUS: Vision-Language Distillated Unseen Synthesizer for Zero-Shot Object Detection".

<img src=".\images\framework.jpg" alt="framework" style="zoom:30%;" />



## 🔨 Requirement

### 1. Environment

- [mmdetection](https://github.com/open-mmlab/mmdetection) we recommend using [Docker 2.0](https://github.com/nasir6/zero_shot_detection/blob/master/Docker.md). Please use the mmdetection codes from this repo.

- The code implementation of our experiments mainly based on Pytorch 1.7.1 , Python 3.7 and CUDA 10.2.

- Install CLIP from the official [CLIP](https://github.com/openai/CLIP) repository.

- ```shell
  git clone https://github.com/Xxxnh/VLDUS.git
  cd VLDUS
  conda env create -f environment.yml
  ```

### 2. Datasets

Download the datasets from the links below and place them in folders named after each dataset within the `mmdetection/data/`.

- [MSCOCO 2014](https://cocodataset.org/#download) 
- [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html) / [2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html)
- [DIOR](https://aistudio.baidu.com/datasetdetail/53045)



## 🏰 Metrics, Pretrained Models and Semantic Embeddings

- We provide metrics (mAP), pretrained models, and semantic embeddings for different datasets in the table below. 
- "ckpt" refers to the pretrained weights for Faster-RCNN, "cls-ckpt" refers to the pretrained weights for classifier, and "se" refers to the semantic embedding files used to train the GAN and classifier.
- Please download and put the "ckpt" in the `mmdetection/work_dir/`, "cls-ckpt" in the `checkpoints/`, and "se" in the respective directories: `MSCOCO/`, `VOC/`, or `DIOR/`.

| Dataset    | Split | ZSD  | Seen | Unseen | HM   | ckpt                                                         | cls-ckpt                                                     | se                                                           |
| :--------- | :---- | :--- | :--- | :----- | :--- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| MSCOCO     | 48/17 | 17.7 | 30.5 | 17.3   | 22.1 | [ckpt-4817](https://drive.google.com/file/d/1K-yn8oCHgISTzqm9IKtvNzERYF8160G7/view?usp=drive_link) | [clsckpt-4817](https://drive.google.com/file/d/1IVl2_2FWA1vlGMOnYsVKq5Ppjgbdoju9/view?usp=drive_link) | [se-coco](https://drive.google.com/file/d/13BBdoH5VCwLjqgar-_Ap-YCuOXWgovm5/view?usp=drive_link) |
| MSCOCO     | 65/15 | 26.3 | 38.6 | 26.1   | 31.2 | [ckpt-6515](https://drive.google.com/file/d/1-8MYtXLErT7fbwrhJR47IpN0tDZhiaS2/view?usp=drive_link) | [clsckpt-6515](https://drive.google.com/file/d/1IbPMO2jirJU0hKInJk5TGFMomdJ-X3P2/view?usp=drive_link) | [se-coco](https://drive.google.com/file/d/13BBdoH5VCwLjqgar-_Ap-YCuOXWgovm5/view?usp=drive_link) |
| PASCAL VOC | 16/4  | 66.3 | 54.2 | 46.3   | 49.9 | [ckpt-voc](https://drive.google.com/file/d/1QCSVq4iGMNMljHRjdTDg1dCw8uGFmumc/view?usp=drive_link) | [clsckpt-voc](https://drive.google.com/file/d/1IBoc18vBceS3A3qj8bXz5K2BAUFeW4oU/view?usp=drive_link) | [se-voc](https://drive.google.com/file/d/13wfSPXQE8hAnL2vEdPWxWTSqq-c2Z7iy/view?usp=drive_link) |
| DIOR       | 16/4  | 12.3 | 22.9 | 8.3    | 12.2 | [ckpt-dior](https://drive.google.com/file/d/19MTxEVt3UyJMqf_HBNyCKhrgl4XddOe3/view?usp=drive_link) | [clsckpt-dior](https://drive.google.com/file/d/1-parapvGsxkoEVhtEfhin6Lpxp_nZMS0/view?usp=drive_link) | [se-dior](https://drive.google.com/file/d/1Q_-oo11JRCtWsK9vR4IrozkgVBo6JihE/view?usp=drive_link) |



## 🚀 Reproducing

- The following scripts correspond to different steps in the pipeline for the MSCOCO 65/15 dataset. Please refer to the respective files for additional arguments. Before running the scripts, make sure to set the dataset, backbone paths and other parameters in the configuration files in `mmdetection/configs/`.
- Weights for [ResNet101](https://drive.google.com/file/d/1g3UXPw-_K3na7acQGZlhjgQPjXz_FNnX/view?usp=sharing) trained excluding overlapping unseen classes from ImageNet.
- The pipeline for other splits and datasets follows a similar process. For specific details, please refer to the code.

#### 1. Train Object Detector on Seen Classes

```shell
cd mmdetection
./tools/dist_train.sh [path to config file] [num of gpus] --validate

# example

./tools/dist_train.sh configs/faster_rcnn_r101_fpn_1x.py 4 --validate
```

#### 2. Extract Features

```shell
cd mmdetection

# extract seen classes features to train Synthesizer and unseen class features for cross validation
python tools/zero_shot_utils.py [path to config file] --classes [seen, unseen] --load_from [detector checkpoint path] --save_dir [path to save features] --data_split [train, test]

# example to extract training features for seen classes

python tools/zero_shot_utils.py configs/faster_rcnn_r101_fpn_1x.py --classes seen --load_from ./work_dirs/coco2014/coco6515/epoch_12.pth --save_dir ./feature/coco14/coco6515 --data_split train

# example to extract test features for unseen classes

python tools/zero_shot_utils.py configs/faster_rcnn_r101_fpn_1x.py --classes unseen --load_from ./work_dirs/coco2014/coco6515/epoch_12.pth --save_dir ./feature/coco14/coco6515 --data_split test
```

#### 3. Train Unseen Classifier and Synthesizer

```shell
# remember to modify parameters such as the dataset settings, 
# the path to th semantic embedding numpy file 
# and the path to save the trained classifier best checkpoint.

python train_unseen_classifier.py

# remember to modify parameters such as the paths to the extracted features,
# labels, and model checkpoints.

./script/train_coco_generator_65_15.sh
```

#### 4. Inference and Evaluate

```shell
cd mmdetection
./tools/dist_test.sh [path to config file] [detector checkpoint path for seen classes] [num of gpus] --dataset [coco, voc, dior] --out [file name to save detection results] [--zsd, --gzsd] --syn_weights [path to synthesized classifier checkpoint]

# ZSD example 

./tools/dist_test.sh configs/faster_rcnn_r101_fpn_1x.py work_dirs/coco2014/coco6515/epoch_12.pth 4 --dataset coco --out coco_results.pkl --zsd --syn_weights ../checkpoints/coco_65_15/classifier_best.pt

# GZSD example 

./tools/dist_test.sh configs/faster_rcnn_r101_fpn_1x.py work_dirs/coco2014/coco6515/epoch_12.pth 4 --dataset coco --out coco_results.pkl --gzsd --syn_weights ../checkpoints/coco_65_15/classifier_best.pt
```



## 🍔 Qualitative Results

<img src=".\images\result.jpg" alt="result" style="zoom:30%;" />


