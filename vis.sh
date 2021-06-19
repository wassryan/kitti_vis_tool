# conda activate kitti_vis
# python kitti_visualize.py --img_fov --vis --show_image_with_boxes --dir ~/Project/KITTI_VIS/scene1 --show_lidar_topview_with_boxes -p

# apply prediction on visualization & merge image to video
ROOT="~/Project/KITTI_VIS/scene2"
TRAIN=$ROOT"/training"
python kitti_visualize.py --img_fov --vis --show_image_with_boxes --dir $ROOT --show_lidar_topview_with_boxes -p
python img2video.py -d $TRAIN -s 0 -e 203 -save $TRAIN