# SuperPoint-SLAM

This repository was forked from ORB-SLAM2 https://github.com/raulmur/ORB_SLAM2.  SuperPoint-SLAM is a modified version of ORB-SLAM2 which use SuperPoint as its feature detector and descriptor. The pre-trained model of SuperPoint  come from https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork.


### Related Publications:

[Monocular] Raúl Mur-Artal, J. M. M. Montiel and Juan D. Tardós. **ORB-SLAM: A Versatile and Accurate Monocular SLAM System**. *IEEE Transactions on Robotics,* vol. 31, no. 5, pp. 1147-1163, 2015. (**2015 IEEE Transactions on Robotics Best Paper Award**). **[PDF](http://webdiis.unizar.es/~raulmur/MurMontielTardosTRO15.pdf)**.

[Stereo and RGB-D] Raúl Mur-Artal and Juan D. Tardós. **ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras**. *IEEE Transactions on Robotics,* vol. 33, no. 5, pp. 1255-1262, 2017. **[PDF](https://128.84.21.199/pdf/1610.06475.pdf)**.

[DBoW2 Place Recognizer] Dorian Gálvez-López and Juan D. Tardós. **Bags of Binary Words for Fast Place Recognition in Image Sequences**. *IEEE Transactions on Robotics,* vol. 28, no. 5, pp.  1188-1197, 2012. **[PDF](http://doriangalvez.com/php/dl.php?dlp=GalvezTRO12.pdf)**

[SuperPoint] DeTone D, Malisiewicz T, Rabinovich A. **Superpoint: Self-supervised interest point detection and description**. InProceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops 2018 (pp. 224-236). **[PDF](https://arxiv.org/abs/1712.07629)**

# 1. License (inherited from ORB-SLAM2)

ORB-SLAM2 is released under a [GPLv3 license](https://github.com/raulmur/ORB_SLAM2/blob/master/License-gpl.txt). For a list of all code/library dependencies (and associated licenses), please see [Dependencies.md](https://github.com/raulmur/ORB_SLAM2/blob/master/Dependencies.md).

For a closed-source version of ORB-SLAM2 for commercial purposes, please contact the authors: orbslam (at) unizar (dot) es.

If you use ORB-SLAM2 (Monocular) in an academic work, please cite:

    @article{murTRO2015,
      title={{ORB-SLAM}: a Versatile and Accurate Monocular {SLAM} System},
      author={Mur-Artal, Ra\'ul, Montiel, J. M. M. and Tard\'os, Juan D.},
      journal={IEEE Transactions on Robotics},
      volume={31},
      number={5},
      pages={1147--1163},
      doi = {10.1109/TRO.2015.2463671},
      year={2015}
     }

if you use ORB-SLAM2 (Stereo or RGB-D) in an academic work, please cite:

    @article{murORB2,
      title={{ORB-SLAM2}: an Open-Source {SLAM} System for Monocular, Stereo and {RGB-D} Cameras},
      author={Mur-Artal, Ra\'ul and Tard\'os, Juan D.},
      journal={IEEE Transactions on Robotics},
      volume={33},
      number={5},
      pages={1255--1262},
      doi = {10.1109/TRO.2017.2705103},
      year={2017}
     }

# 2. Prerequisites
We have tested the library in **Ubuntu 12.04**, **14.04** and **16.04**, but it should be easy to compile in other platforms. A powerful computer (e.g. i7) will ensure real-time performance and provide more stable and accurate results.

## C++11 or C++0x Compiler
We use the new thread and chrono functionalities of C++11.

## Pangolin
We use [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization and user interface. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

## OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. **Required at leat 2.4.3. Tested with OpenCV 2.4.11 and OpenCV 3.2**.

## Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

## DBoW3 and g2o (Included in Thirdparty folder)
We use modified versions of [DBoW3](https://github.com/rmsalinas/DBow3) (instead of DBoW2) library to perform place recognition and [g2o](https://github.com/RainerKuemmerle/g2o) library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the *Thirdparty* folder.

## Libtorch

We use Pytorch C++ API to implement SuperPoint model. It can be built as follows:

``` shell
git clone --recursive -b v1.0.1 https://github.com/pytorch/pytorch
cd pytorch && mkdir build && cd build
python ../tools/build_libtorch.py
```

It may take quite a long time to download and build. Please wait with patience.

**NOTE**: Do not use the pre-built package in the official website, it would cause some errors.

# 3. Building SuperPoint-SLAM library and examples

Clone the repository:
```
git clone https://github.com/KinglittleQ/SuperPoint_SLAM.git SuperPoint_SLAM
```

We provide a script `build.sh` to build the *Thirdparty* libraries and *SuperPoint_SLAM*. Please make sure you have **installed all required dependencies** (see section 2). Execute:
```
cd SuperPoint_SLAM
chmod +x build.sh
./build.sh
```

This will create **libSuerPoint_SLAM.so**  at *lib* folder and the executables **mono_tum**, **mono_kitti**, **mono_euroc** in *Examples* folder.

**TIPS:**

If cmake cannot find some package such as OpenCV or EIgen3, try to set XX_DIR which contain XXConfig.cmake manually. Add the following statement into `CMakeLists.txt`  before `find_package(XX)`:

``` cmake
set(XX_DIR "your_path")
# set(OpenCV_DIR "usr/share/OpenCV")
# set(Eigen3_DIR "usr/share/Eigen3")
```

# 4. Download Vocabulary

You can download the vocabulary from [google drive](https://drive.google.com/file/d/1p1QEXTDYsbpid5ELp3IApQ8PGgm_vguC/view?usp=sharing) or [BaiduYun](https://pan.baidu.com/s/1fygQil78GpoPm0zoi6BMng) (code: de3g). And then put it into `Vocabulary` directory. The vocabulary was trained on [Bovisa_2008-09-01](http://www.rawseeds.org/rs/datasets/view//7) using DBoW3 library. Branching factor k and depth levels L are set to 5 and 10 respectively.

# 5. Monocular Examples

## KITTI Dataset  

1. Download the dataset (grayscale images) from http://www.cvlibs.net/datasets/kitti/eval_odometry.php 

2. Execute the following command. Change `KITTIX.yaml`by KITTI00-02.yaml, KITTI03.yaml or KITTI04-12.yaml for sequence 0 to 2, 3, and 4 to 12 respectively. Change `PATH_TO_DATASET_FOLDER` to the uncompressed dataset folder. Change `SEQUENCE_NUMBER` to 00, 01, 02,.., 11. 
```
./Examples/Monocular/mono_kitti Vocabulary/ORBvoc.txt Examples/Monocular/KITTIX.yaml PATH_TO_DATASET_FOLDER/dataset/sequences/SEQUENCE_NUMBER
```

# 6. SLAM and Localization Modes
You can change between the *SLAM* and *Localization mode* using the GUI of the map viewer.

### SLAM Mode
This is the default mode. The system runs in parallal three threads: Tracking, Local Mapping and Loop Closing. The system localizes the camera, builds new map and tries to close loops.

### Localization Mode
This mode can be used when you have a good map of your working area. In this mode the Local Mapping and Loop Closing are deactivated. The system localizes the camera in the map (which is no longer updated), using relocalization if needed. 

# TODO lists


- [x] Upload the vocabulary of SuperPoint
- [ ] Clean the code
- [ ] Stereo, RGBD

