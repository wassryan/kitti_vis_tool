# KITTI Object data transformation and visualization



## Dataset

Download the data (calib, image\_2, label\_2, velodyne) from [Kitti Object Detection Dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) and place it in your data folder at `kitti/object`


The folder structure is as following:
```
kitti
    object
        testing
            calib
            image_2
            label_2
            velodyne
        training
            calib
            image_2
            label_2
            velodyne
```

## Install locally on a Ubuntu 16.04 PC with GUI
- start from a new conda enviornment:
```
(base)$ conda create -n kitti_vis python=3.7 # vtk does not support python 3.8
(base)$ conda activate kitti_vis
```
- opencv, pillow, scipy, matplotlib
```
(kitti_vis)$ pip install opencv-python pillow scipy matplotlib
```
- install mayavi from conda-forge, this installs vtk and pyqt5 automatically
```
(kitti_vis)$ conda install mayavi -c conda-forge
```
- test installation
```
(kitti_vis)$ python kitti_object.py --show_lidar_with_depth --img_fov --const_box --vis
```

**Note: the above installation has been tested not work on MacOS.**

## Visualization

1. 3D boxes on LiDar point cloud in volumetric mode
2. 2D and 3D boxes on Camera image
3. 2D boxes on LiDar Birdview
4. LiDar data on Camera image


refer to **vis.sh**
