""" Helper class and functions for loading KITTI objects

Author: Charles R. Qi
Date: September 2017
"""
from __future__ import print_function
from kitti_util import lidar_to_top_coords

import os
import sys
import numpy as np
import cv2
import mayavi.mlab as mlab
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "mayavi"))
import kitti_util as utils
import argparse

try:
    raw_input  # Python 2
except NameError:
    raw_input = input  # Python 3
import math
cbox = np.array([[0, 70.4], [-40, 40], [-3, 1]])

BEV_dir = None
IMG_dir = None
save_idx = 0
save_idx2 = 0

class kitti_object(object):
    """Load and parse object data into a usable format."""

    def __init__(self, root_dir, split="training", args=None):
        """root_dir contains training and testing folders"""
        self.root_dir = root_dir
        self.split = split
        print(root_dir, split)
        self.split_dir = os.path.join(root_dir, split)

        if split == "training":
            self.num_samples = 7481
        elif split == "testing":
            self.num_samples = 7518
        else:
            print("Unknown split: %s" % (split))
            exit(-1)

        lidar_dir = "velodyne"
        depth_dir = "depth"
        pred_dir = "pred"
        if args is not None:
            lidar_dir = args.lidar
            depth_dir = args.depthdir
            pred_dir = args.preddir

        self.image_dir = os.path.join(self.split_dir, "image_2")
        self.label_dir = os.path.join(self.split_dir, "label_2")
        self.calib_dir = os.path.join(self.split_dir, "calib")

        self.depthpc_dir = os.path.join(self.split_dir, "depth_pc")
        self.lidar_dir = os.path.join(self.split_dir, lidar_dir)
        self.depth_dir = os.path.join(self.split_dir, depth_dir)
        self.pred_dir = os.path.join(self.split_dir, pred_dir)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert idx < self.num_samples
        img_filename = os.path.join(self.image_dir, "%06d.png" % (idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx, dtype=np.float32, n_vec=4):
        assert idx < self.num_samples
        lidar_filename = os.path.join(self.lidar_dir, "%06d.bin" % (idx))
        print(lidar_filename)
        return utils.load_velo_scan(lidar_filename, dtype, n_vec) # 从bin中读取点云数据并reshape成(N,n_vec=4)

    def get_calibration(self, idx):
        assert idx < self.num_samples
        calib_filename = os.path.join(self.calib_dir, "%06d.txt" % (idx))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert idx < self.num_samples and self.split == "training"
        label_filename = os.path.join(self.label_dir, "%06d.txt" % (idx))
        return utils.read_label(label_filename)

    def get_pred_objects(self, idx):
        assert idx < self.num_samples
        pred_filename = os.path.join(self.pred_dir, "%06d.txt" % (idx))
        is_exist = os.path.exists(pred_filename)
        if is_exist:
            return utils.read_label(pred_filename)
        else:
            return None

    def get_depth(self, idx):
        assert idx < self.num_samples
        img_filename = os.path.join(self.depth_dir, "%06d.png" % (idx))
        return utils.load_depth(img_filename)

    def get_depth_image(self, idx):
        assert idx < self.num_samples
        img_filename = os.path.join(self.depth_dir, "%06d.png" % (idx))
        return utils.load_depth(img_filename)

    def get_depth_pc(self, idx):
        assert idx < self.num_samples
        lidar_filename = os.path.join(self.depthpc_dir, "%06d.bin" % (idx))
        is_exist = os.path.exists(lidar_filename)
        if is_exist:
            return utils.load_velo_scan(lidar_filename), is_exist
        else:
            return None, is_exist
        # print(lidar_filename, is_exist)
        # return utils.load_velo_scan(lidar_filename), is_exist

    def get_top_down(self, idx):
        pass

    def isexist_pred_objects(self, idx):
        assert idx < self.num_samples and self.split == "training"
        pred_filename = os.path.join(self.pred_dir, "%06d.txt" % (idx))
        return os.path.exists(pred_filename)

    def isexist_depth(self, idx):
        assert idx < self.num_samples and self.split == "training"
        depth_filename = os.path.join(self.depth_dir, "%06d.txt" % (idx))
        return os.path.exists(depth_filename)


class kitti_object_video(object):
    """ Load data for KITTI videos """

    def __init__(self, img_dir, lidar_dir, calib_dir):
        self.calib = utils.Calibration(calib_dir, from_video=True)
        self.img_dir = img_dir
        self.lidar_dir = lidar_dir
        self.img_filenames = sorted(
            [os.path.join(img_dir, filename) for filename in os.listdir(img_dir)]
        )
        self.lidar_filenames = sorted(
            [os.path.join(lidar_dir, filename) for filename in os.listdir(lidar_dir)]
        )
        print(len(self.img_filenames))
        print(len(self.lidar_filenames))
        # assert(len(self.img_filenames) == len(self.lidar_filenames))
        self.num_samples = len(self.img_filenames)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert idx < self.num_samples
        img_filename = self.img_filenames[idx]
        return utils.load_image(img_filename)

    def get_lidar(self, idx):
        assert idx < self.num_samples
        lidar_filename = self.lidar_filenames[idx]
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, unused):
        return self.calib


def viz_kitti_video():
    video_path = os.path.join(ROOT_DIR, "dataset/2011_09_26/")
    dataset = kitti_object_video(
        os.path.join(video_path, "2011_09_26_drive_0023_sync/image_02/data"),
        os.path.join(video_path, "2011_09_26_drive_0023_sync/velodyne_points/data"),
        video_path,
    )
    print(len(dataset))
    for _ in range(len(dataset)):
        img = dataset.get_image(0)
        pc = dataset.get_lidar(0)
        cv2.imshow("video", img)
        draw_lidar(pc)
        raw_input()
        pc[:, 0:3] = dataset.get_calibration().project_velo_to_rect(pc[:, 0:3])
        draw_lidar(pc)
        raw_input()
    return


def show_image_with_boxes(img, objects, pred_objects, calib, show3d=True, depth=None, data_idx=None, warn_list=[], warn_img_list=[]):
    global save_idx
    """ Show image with 2D bounding boxes """
    img1 = np.copy(img)  # for 2d bbox
    img2 = np.copy(img)  # for 3d bbox
    #img3 = np.copy(img)  # for 3d bbox
    #TODO: change the color of boxes
    # show gt box
    for ix, obj in enumerate(objects):

        # from ipdb import set_trace; set_trace()
        if obj.type == "DontCare":
            continue
        if obj.type == "Car":
            cv2.rectangle(
            img1,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            (0, 255, 0),
            2,
        )
        if obj.type == "Pedestrian":
            cv2.rectangle(
            img1,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            (255, 255, 0),
            2,
        )
        if obj.type == "Cyclist":
            cv2.rectangle(
            img1,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            (0, 255, 255),
            2,
        )
        box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P) # project 3d into image plane
        # set_trace()
        if obj.type == "Car":
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 0))
        elif obj.type == "Pedestrian":
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(255, 255, 0))
        elif obj.type == "Cyclist":
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 255))
    # show pred box
    is_warn_img = False
    if data_idx in warn_img_list:
        is_warn_img = True
    for ix, obj in enumerate(pred_objects):
        is_warn = False
        if ix in warn_list:
            is_warn = True

        if obj.type == "DontCare":
            continue
        if obj.type == "Car":
            cv2.rectangle(
            img1,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            (0, 255, 0),
            2,
        )
        if obj.type == "Pedestrian":
            cv2.rectangle(
            img1,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            (255, 255, 0),
            2,
        )
        if obj.type == "Cyclist":
            cv2.rectangle(
            img1,
            (int(obj.xmin), int(obj.ymin)),
            (int(obj.xmax), int(obj.ymax)),
            (0, 255, 255),
            2,
        )
        box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P) # project 3d into image plane
        # set_trace()
        if obj.type == "Car":
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 0, 255), is_warn=is_warn)
        elif obj.type == "Pedestrian":
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(255, 255, 0))
        elif obj.type == "Cyclist":
            img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, color=(0, 255, 255))

        if is_warn:
            cv2.putText(img2, "Warning!!", (img.shape[1]//2-200,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)

        # project
        # box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        # box3d_pts_32d = utils.box3d_to_rgb_box00(box3d_pts_3d_velo)
        # box3d_pts_32d = calib.project_velo_to_image(box3d_pts_3d_velo)
        # img3 = utils.draw_projected_box3d(img3, box3d_pts_32d)
    # print("img1:", img1.shape)
    # cv2.imshow("2dbox", img1)
    # print("img3:",img3.shape)
    # Image.fromarray(img3).show()
    show3d = True
    if show3d:
        # print("img2:",img2.shape)
        cv2.imshow("3dbox", img2)
        cv2.imwrite(os.path.join(IMG_dir, "{:06d}.png".format(save_idx)), img2)
        if is_warn_img:
            cv2.imwrite(os.path.join(IMG_dir, "{:06d}.png".format(save_idx+1)), img2)
            cv2.imwrite(os.path.join(IMG_dir, "{:06d}.png".format(save_idx+2)), img2)
            save_idx += 2
    if depth is not None:
        cv2.imshow("depth", depth)
    save_idx += 1
    return img1, img2


def get_lidar_in_image_fov(
    pc_velo, calib, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0
):
    """ Filter lidar points, keep those in image FOV """
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (
        (pts_2d[:, 0] < xmax)
        & (pts_2d[:, 0] >= xmin)
        & (pts_2d[:, 1] < ymax)
        & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def get_lidar_index_in_image_fov(
    pc_velo, calib, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0
):
    """ Filter lidar points, keep those in image FOV """
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (
        (pts_2d[:, 0] < xmax)
        & (pts_2d[:, 0] >= xmin)
        & (pts_2d[:, 1] < ymax)
        & (pts_2d[:, 1] >= ymin)
    )
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    return fov_inds


def depth_region_pt3d(depth, obj):
    b = obj.box2d
    # depth_region = depth[b[0]:b[2],b[2]:b[3],0]
    pt3d = []
    # import pdb; pdb.set_trace()
    for i in range(int(b[0]), int(b[2])):
        for j in range(int(b[1]), int(b[3])):
            pt3d.append([j, i, depth[j, i]])
    return np.array(pt3d)


def get_depth_pt3d(depth):
    pt3d = []
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            pt3d.append([i, j, depth[i, j]])
    return np.array(pt3d)


def show_lidar_with_depth(
    pc_velo,
    objects,
    calib,
    fig,
    img_fov=False,
    img_width=None,
    img_height=None,
    objects_pred=None,
    depth=None,
    cam_img=None,
    constraint_box=False,
    pc_label=False,
    save=False,
):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """
    if "mlab" not in sys.modules:
        import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    print(("All point num: ", pc_velo.shape[0]))
    if img_fov:
        pc_velo_index = get_lidar_index_in_image_fov(
            pc_velo[:, :3], calib, 0, 0, img_width, img_height
        ) # 获取是front view之前的point cloud的index
        pc_velo = pc_velo[pc_velo_index, :]
        print(("FOV point num: ", pc_velo.shape))
    print("pc_velo", pc_velo.shape)
    draw_lidar(pc_velo, fig=fig, pc_label=pc_label)

    # Draw depth
    if depth is not None:
        depth_pc_velo = calib.project_depth_to_velo(depth, constraint_box)

        indensity = np.ones((depth_pc_velo.shape[0], 1)) * 0.5
        depth_pc_velo = np.hstack((depth_pc_velo, indensity))
        print("depth_pc_velo:", depth_pc_velo.shape)
        print("depth_pc_velo:", type(depth_pc_velo))
        print(depth_pc_velo[:5])
        draw_lidar(depth_pc_velo, fig=fig, pts_color=(1, 1, 1))

        if save:
            data_idx = 0
            vely_dir = "data/object/training/depth_pc"
            save_filename = os.path.join(vely_dir, "%06d.bin" % (data_idx))
            print(save_filename)
            # np.save(save_filename+".npy", np.array(depth_pc_velo))
            depth_pc_velo = depth_pc_velo.astype(np.float32)
            depth_pc_velo.tofile(save_filename)

    # color = (0, 1, 0)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        # Draw 3d bounding box
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        print("box3d_pts_3d_velo:")
        print(box3d_pts_3d_velo)

        #TODO: change the color of boxes
        if obj.type == "Car":
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=(0,1,0), label=obj.type)
        elif obj.type == "Pedestrian":
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=(0,1,1), label=obj.type)
        elif obj.type == "Cyclist":
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=(1,1,0), label=obj.type)


    if objects_pred is not None:
        color = (1, 0, 0)
        for obj in objects_pred:
            if obj.type == "DontCare":
                continue
            # Draw 3d bounding box
            _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            print("box3d_pts_3d_velo:")
            print(box3d_pts_3d_velo)
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig, color=color)
            # Draw heading arrow
            _, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1, y1, z1 = ori3d_pts_3d_velo[0, :]
            x2, y2, z2 = ori3d_pts_3d_velo[1, :]
            # print("Draw arrow!!!!")
            mlab.plot3d(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                color=color,
                tube_radius=None,
                line_width=1,
                figure=fig,
            )
    # mlab.show(1)


def box_min_max(box3d):
    box_min = np.min(box3d, axis=0)
    box_max = np.max(box3d, axis=0)
    return box_min, box_max


def get_velo_whl(box3d, pc):
    bmin, bmax = box_min_max(box3d)
    ind = np.where(
        (pc[:, 0] >= bmin[0])
        & (pc[:, 0] <= bmax[0])
        & (pc[:, 1] >= bmin[1])
        & (pc[:, 1] <= bmax[1])
        & (pc[:, 2] >= bmin[2])
        & (pc[:, 2] <= bmax[2])
    )[0]
    # print(pc[ind,:])
    if len(ind) > 0:
        vmin, vmax = box_min_max(pc[ind, :])
        return vmax - vmin
    else:
        return 0, 0, 0, 0


def stat_lidar_with_boxes(pc_velo, objects, calib):
    """ Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) """

    # print(('All point num: ', pc_velo.shape[0]))

    # draw_lidar(pc_velo, fig=fig)
    # color=(0,1,0)
    for obj in objects:
        if obj.type == "DontCare":
            continue
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        v_l, v_w, v_h, _ = get_velo_whl(box3d_pts_3d_velo, pc_velo)
        print("%.4f %.4f %.4f %s" % (v_w, v_h, v_l, obj.type))


def show_lidar_topview_with_boxes(pc_velo, objects, calib, objects_pred=None, data_idx=None):
    global save_idx2
    """ top_view image"""
    # print('pc_velo shape: ',pc_velo.shape)
    top_view = utils.lidar_to_top(pc_velo)
    top_image = utils.draw_top_image(top_view)
    print("top_image:", top_image.shape)
    # gt

    def bbox3d(obj):
        _, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        return box3d_pts_3d_velo

    boxes3d = [bbox3d(obj) for obj in objects if obj.type != "DontCare"]
    gt = np.array(boxes3d)
    # print("box2d BV:",boxes3d)
    lines = [obj.type for obj in objects if obj.type != "DontCare"]
    top_image = utils.draw_box3d_on_top(
        top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=True
    )
    # pred
    total_flag = False
    warn_list = []
    warn_img_list = []
    if objects_pred is not None:
        boxes3d = [bbox3d(obj) for obj in objects_pred if obj.type != "DontCare"]
        gt = np.array(boxes3d)
        lines = [obj.type for obj in objects_pred if obj.type != "DontCare"]
        # from ipdb import set_trace; set_trace()
        top_image = utils.draw_box3d_on_top(
            top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=False
        )
        # @zk add
        center_gt = np.zeros((gt.shape[0], 4)) # (N,4) 4-> x,y,z,distance
        for i in range(gt.shape[0]):
            per_gt = gt[i]
            # print(per_gt)
            center_x = (per_gt[0,0] + per_gt[1,0]) / 2.
            center_y = (per_gt[0,1] + per_gt[3,1]) / 2.
            center_z = (per_gt[0,2] + per_gt[4,2]) / 2.
            # print(center_x, center_y, center_z)
            center_gt[i,0] = center_x
            center_gt[i,1] = center_y
            center_gt[i,2] = center_z
            center_gt[i,3] = np.sqrt(center_x ** 2 + center_y ** 2) # physical distance

            def draw_center_on_top(image, center):
                flag = False
                origin_y, origin_x = image.shape[0], image.shape[1]//2
                u, v = lidar_to_top_coords(center[0], center[1])
                
                def cal_angle(v1, v2):
                    """
                    v1/v2: [x1,y1,x2,y2]
                    """
                    dx1 = v1[2] - v1[0]
                    dy1 = v1[3] - v1[1]
                    dx2 = v2[2] - v2[0]
                    dy2 = v2[3] - v2[1]
                    angle1 = math.atan2(dy1, dx1)
                    angle1 = int(angle1 * 180/math.pi)
                    # print(angle1)
                    angle2 = math.atan2(dy2, dx2)
                    angle2 = int(angle2 * 180/math.pi)
                    # print(angle2)
                    if angle1*angle2 >= 0:
                        included_angle = abs(angle1-angle2)
                    else:
                        included_angle = abs(angle1) + abs(angle2)
                        if included_angle > 180:
                            included_angle = 360 - included_angle
                    return included_angle

                v1 = [origin_x, origin_y, u, v]
                v2 = [image.shape[1]//2, image.shape[0], image.shape[1], image.shape[0]]
                angle = cal_angle(v1, v2)           
                cv2.putText(image, str(angle), (u+5,v+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 0, cv2.LINE_AA)
                
                if angle >= 88 and angle <= 92 and center[3] < 14:
                    flag = True
                if flag:
                    cv2.circle(image, (u, v), 2, (255, 255, 255), 2)
                    cv2.line(image, (origin_x, origin_y), (u,v), (0,0,255), 2, cv2.LINE_AA)
                    cv2.putText(image, "{}m".format(str(int(center[3]))), (u+5,v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 0, cv2.LINE_AA)
                else:
                    cv2.circle(image, (u, v), 2, (255, 255, 255), 2)
                    cv2.line(image, (origin_x, origin_y), (u,v), (255,255,255), 1, cv2.LINE_AA)
                    cv2.putText(image, "{}m".format(str(int(center[3]))), (u+5,v-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 0, cv2.LINE_AA)

                return image, flag

            top_image, flag = draw_center_on_top(top_image, center_gt[i])
                
            if flag:
                total_flag = True
                warn_list.append(i)

    if total_flag:
        cv2.rectangle(top_image, (0, 0), (top_image.shape[1], top_image.shape[0]), (0, 0, 255), 3)
        warn_img_list.append(data_idx)
    cv2.imshow("top_image", top_image)
    cv2.imwrite(os.path.join(BEV_dir, "{:06d}.png".format(save_idx2)), top_image)
    if total_flag:
        cv2.imwrite(os.path.join(BEV_dir, "{:06d}.png".format(save_idx2+1)), top_image)
        cv2.imwrite(os.path.join(BEV_dir, "{:06d}.png".format(save_idx2+2)), top_image)
        save_idx2 = save_idx2 + 2
    save_idx2 += 1
    return warn_list, warn_img_list


def dataset_viz(root_dir, args):
    global BEV_dir, IMG_dir
    BEV_dir = os.path.join(root_dir, args.split, "bev")
    IMG_dir = os.path.join(root_dir, args.split, "img")

    if not os.path.exists(BEV_dir):
        os.makedirs(BEV_dir)
    if not os.path.exists(IMG_dir):
        os.makedirs(IMG_dir)

    dataset = kitti_object(root_dir, split=args.split, args=args)
    ## load 2d detection results
    #objects2ds = read_det_file("box2d.list")

    if args.show_lidar_with_depth:
        # import mayavi.mlab as mlab
        fig = None
        # fig = mlab.figure(
        #     figure='point cloud', bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500)
        # )
        # fig = mlab.figure(
            # figure='point cloud', bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(600, 550)
        # )

    for data_idx in range(len(dataset)):
        if args.ind > 0:
            data_idx = args.ind
        # Load data from dataset
        if args.split == "training":
            objects = dataset.get_label_objects(data_idx)
        else:
            objects = []
        #objects2d = objects2ds[data_idx]

        objects_pred = None
        if args.pred:
            # if not dataset.isexist_pred_objects(data_idx):
            #    continue
            objects_pred = dataset.get_pred_objects(data_idx)
            if objects_pred == None:
                continue
        if objects_pred == None:
            print("no pred file")
            # objects_pred[0].print_object()

        n_vec = 4
        if args.pc_label:
            n_vec = 5

        dtype = np.float32
        if args.dtype64:
            dtype = np.float64
        pc_velo = dataset.get_lidar(data_idx, dtype, n_vec)[:, 0:n_vec] # (N,4)
        calib = dataset.get_calibration(data_idx)
        img = dataset.get_image(data_idx)
        img_height, img_width, _ = img.shape
        print(data_idx, "image shape: ", img.shape)
        # print(data_idx, "velo  shape: ", pc_velo.shape)
        if args.depth:
            depth, _ = dataset.get_depth(data_idx)
            # print(data_idx, "depth shape: ", depth.shape)
        else:
            depth = None

        # depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        # depth_height, depth_width, depth_channel = img.shape

        # print(('Image shape: ', img.shape))
        # from ipdb import set_trace; set_trace()
        if args.stat:
            stat_lidar_with_boxes(pc_velo, objects, calib)
            continue
        # print("======== Objects in Ground Truth ========")
        n_obj = 0
        for obj in objects:
            if obj.type != "DontCare":
                # print("=== {} object ===".format(n_obj + 1))
                obj.print_object()
                n_obj += 1

        # Draw 3d box in LiDAR point cloud
        if args.show_lidar_topview_with_boxes:
            # Draw lidar top view
            warn_list, warn_img_list = show_lidar_topview_with_boxes(pc_velo, objects, calib, objects_pred, data_idx)

        if args.show_image_with_boxes:
            # Draw 2d and 3d boxes on image
            show_image_with_boxes(img, objects, objects_pred, calib, True, depth, data_idx, warn_list, warn_img_list)
        if args.show_lidar_with_depth:
            # Draw 3d box in LiDAR point cloud
            show_lidar_with_depth(
                pc_velo,
                objects,
                calib,
                fig,
                args.img_fov,
                img_width,
                img_height,
                objects_pred,
                depth,
                img,
                constraint_box=args.const_box,
                save=args.save_depth,
                pc_label=args.pc_label,
            )
        # input_str = raw_input()
        if args.show_lidar_with_depth:
            import mayavi.mlab as mlab
            mlab.close()
        # if input_str == "q":
            # break
    assert save_idx == save_idx2
    print("All finish...")



def read_det_file(det_filename):
    """ Parse lines in 2D detection output files """
    #det_id2str = {1: "Pedestrian", 2: "Car", 3: "Cyclist"}
    objects = {}
    with open(det_filename, "r") as f:
        for line in f.readlines():
            obj = utils.Object2d(line.rstrip())
            if obj.img_name not in objects.keys():
                objects[obj.img_name] = []
            objects[obj.img_name].append(obj)
        # objects = [utils.Object2d(line.rstrip()) for line in f.readlines()]

    return objects


if __name__ == "__main__":
    import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    parser = argparse.ArgumentParser(description="KIITI Object Visualization")
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="data/object",
        metavar="N",
        help="input  (default: data/object)",
    )
    parser.add_argument(
        "-i",
        "--ind",
        type=int,
        default=0,
        metavar="N",
        help="input  (default: data/object)",
    )
    parser.add_argument(
        "-p", "--pred", action="store_true", help="show predict results"
    )
    parser.add_argument(
        "-s",
        "--stat",
        action="store_true",
        help=" stat the w/h/l of point cloud in gt bbox",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        help="use training split or testing split (default: training)",
    )
    parser.add_argument(
        "-l",
        "--lidar",
        type=str,
        default="velodyne",
        metavar="N",
        help="velodyne dir  (default: velodyne)",
    )
    parser.add_argument(
        "-e",
        "--depthdir",
        type=str,
        default="depth",
        metavar="N",
        help="depth dir  (default: depth)",
    )
    parser.add_argument(
        "-r",
        "--preddir",
        type=str,
        default="pred",
        metavar="N",
        help="predicted boxes  (default: pred)",
    )
    parser.add_argument("--gen_depth", action="store_true", help="generate depth")
    parser.add_argument("--vis", action="store_true", help="show images")
    parser.add_argument("--depth", action="store_true", help="load depth")
    parser.add_argument("--img_fov", action="store_true", help="front view mapping")
    parser.add_argument("--const_box", action="store_true", help="constraint box")
    parser.add_argument(
        "--save_depth", action="store_true", help="save depth into file"
    )
    parser.add_argument(
        "--pc_label", action="store_true", help="5-verctor lidar, pc with label"
    )
    parser.add_argument(
        "--dtype64", action="store_true", help="for float64 datatype, default float64"
    )

    parser.add_argument(
        "--show_lidar_on_image", action="store_true", help="project lidar on image"
    )
    parser.add_argument(
        "--show_lidar_with_depth",
        action="store_true",
        help="--show_lidar, depth is supported",
    )
    parser.add_argument(
        "--show_image_with_boxes", action="store_true", help="show lidar"
    )
    parser.add_argument(
        "--show_lidar_topview_with_boxes",
        action="store_true",
        help="show lidar topview",
    )
    args = parser.parse_args()
    if args.pred:
        # assert os.path.exists(args.dir + "/" + args.split + "/pred")
        assert os.path.exists(args.dir)

    if args.vis:
        dataset_viz(args.dir, args)

