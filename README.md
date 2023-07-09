# UnderWater Few\-shot
This code is for the paper "Few-shot segmentation and Semantic Segmentation for Underwater
Imagery" (accepted to IROS2023) [[arxiv]()]

Our proposed network architecture for few-shot segmentation
<p align="left">
  <img src="figures/architecture.png" width="100%" height="auto">
</p>

---

## Overview
This code contains our proposed network along with other baselines for comparison.

The experiments are divided into 4 independent groups for cross validation.

---

## Dependencies
python == 3.8,
pytorch == 1.6,

torchvision,
pillow,
opencv-python,
pandas,
matplotlib,
scikit-image

---

## Usage
Training and testing on our proposed network and other baseline networks can be run very easily using this code.  

### Preparation
First, download the code and install dependencies. Then download the dataset and put it in the dataset directory.

You can download dataset from [here]()

Also download the dataset split files from [here]() and put them in dataset_split directory


### Prepare Config Files
Four sample config files are provided in the "experiments" directory. You can modify them according to your requirements.
Please change `LOG_DIR`, `OUTPUT_DIR`, `BASE_DIR`, and dataset `ROOT` according to your folder hierarchy, Here,

- `LOG_DIR` in which file you want to save the logs;
- `OUTPUT_DIR` in which directory you want to save trained models and checkpoint;
- `BASE_DIR` this the absolute path of this repository;
- `ROOT` this is placed under `DATASET` variable in the config file. put the absolute path of dataset directory here;

#### Some Other Configuration switches

- `TRAIN` contains all the switches for initializing model and optimizer;
  - `ARCH` specify which network you want to use (options: 'asgnet', 'FPMMs', 'PAnet', 'PFENet');
  - `N_SHOTS` specify number of shots;
  - `TEST_LABEL_SPLIT_VALUE` specify on which split of dataset you want to run the validation (options: 0,1,2,3);
  - `LAYERS` determines which type of resnet backbone will be used in some networks (50 means resnet50);
  - `RESNET_PRETRAINED_MODEL` specify the path of trained resnet backbone;
  - `VGG_MODEL_PATH` specify the path of trained vgg backbone;
  - `PRETRAINED_MODEL` specify the path of pretrained model path;
- `MODEL` contain model settings;
  - `NAME` specify which network you want to use (options: 'asgnet', 'FPMMs', 'PAnet', 'PFENet');

---

## Cross-validation classes for UnderWater Few-shot
| Dataset Split     | Test class                                                |
|-------------------|-----------------------------------------------------------|
| Split<sup>0</sup> | Crab, Dolphin, Frog, Turtle, Whale                        |
| Split<sup>1</sup> | Nettles, Octopus, Sea Anemone, Shrimp, Stingray           |
| Split<sup>2</sup> | Penguin, Sea Urchin, Seal, Shark, Nudibranch              |
| Split<sup>3</sup> | Crocodile, Otter, Polar Bear, Sea Horse, Star Fish, Squid |


## Network Options

### Training
```
cd tools
python3 train.py --config path_to_config_file
```

#### 1. Train  UWSNet models:

| Network   | Option to be used          |
|-----------|----------------------------|
| PANet     | basic                      |
| UWSNet v1 | eca_net_sup_que            |
| UWSNet v2 | eca_net_sup_que_vgg16      |
| UWSNet v3 | triplet_sup_que            |
| UWSNet v4 | triplet_sup_que_dice       |
| UWSNet v5 | triplet_sup_que_vgg16      |
| UWSNet v6 | triplet_sup_que_vgg16_dice |


Change the `PA_NET_TYPE` value in config file accordingly.



### Inference
If you want to test for a specific saved models, you can use:
```
cd tools
python3 test.py --config path_to_config_file
```

Must set the base architecture (`NAME` under `MODEL`) and type (`PA_NET_TYPE` under `TRAIN`) according to your choice in the config file.
  
### Pre-trained models
- Pre-trained backbones and models can be found in [Google Driver]()
- Download backbones and set the backbone model path 

---

## Citation
Please consider citing the paper if you find it useful:
```
@inproceedings{,
  title={},
  author={},
  booktitle={},
  year={}
}
```

---

## References
The code is based on 

* [FPMMS](https://github.com/Yang-Bob/PMMs)
* [PANet architecture](https://github.com/kaixin96/PANet)
* [ECANet Attention](https://blog.paperspace.com/attention-mechanisms-in-computer-vision-ecanet/)
* [Triplet Attention](https://blog.paperspace.com/triplet-attention-wacv-2021/)
* [Dice Loss](https://github.com/pytorch/pytorch/issues/1249)

Thanks for their great work!

