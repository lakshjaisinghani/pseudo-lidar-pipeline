# BTS Pipeline
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-big-to-small-multi-scale-local-planar/monocular-depth-estimation-on-kitti-eigen)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen?p=from-big-to-small-multi-scale-local-planar) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-big-to-small-multi-scale-local-planar/monocular-depth-estimation-on-nyu-depth-v2)](https://paperswithcode.com/sota/monocular-depth-estimation-on-nyu-depth-v2?p=from-big-to-small-multi-scale-local-planar)

From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation   
[arXiv](https://arxiv.org/abs/1907.10326)  
[Supplementary material](https://arxiv.org/src/1907.10326v4/anc/bts_sm.pdf) 

## Video Demo 
[![Screenshot](https://img.youtube.com/vi/1J-GSb0fROw/maxresdefault.jpg)](https://www.youtube.com/watch?v=1J-GSb0fROw)


## Purpose

This pipeline is used to test the performance of the BTS depth estimation model. It
is subscribed to the `camera/kitti` topic and passes the incoming images to the depth model.
The `mock_publisher.py` is responsible for publishing the images.

![Image of Pipeline in action](https://git.infotech.monash.edu/MonashCAV/software/depth-estimation/tree/development-II/Images/display_test.PNG)

## Start Depth Pipeline

Open a terminal and run:

```shell
$ roscore
```

On a new terminal, run the following code in your terminal once `cd` in the directory.

```shell
$ python2 main.py
```

## Start Mock Publishing

On a new terminal, run the following code in your terminal once `cd` in the directory.

```shell
$ python2 mock_publisher.py config_test.txt
```

Where, the `config_test.txt` contains the path to the KITTI images to be published.

### config_test.txt

The config file provides command line utilities in a file format for ease of use.
The file contains: 

```shell
--encoder densenet161_bts 
--data_path /home/mcav/DATA/kitti_dataset 
--image_path /2011_09_26/2011_09_26_drive_0022_sync/
```