# python img2video.py -d ~/Project/KITTI_VIS/scene1/training/bev -s 0 -e 210 -save ~/Project/KITTI_VIS/scene1/training
import os
import cv2

import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Img to Video")
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        default="~/Project/KITTI_VIS/scene1/training",
        metavar="N",
        help="input  (default: data/object)",
    )
    parser.add_argument("-s", "--start_idx", type=int, default=0)
    parser.add_argument("-e", "--end_idx", type=int, default=50)
    parser.add_argument("-save", "--save_dir", type=str, default="./")
    args = parser.parse_args()

    # concat top view and front view
    BEV_dir = os.path.join(args.dir, 'bev')
    IMG_dir = os.path.join(args.dir, 'img')
    print(BEV_dir)
    print(IMG_dir)

    oh, ow = (384, 230)
    bev = cv2.imread(os.path.join(BEV_dir, "{:06d}.png".format(args.start_idx)))
    bev = cv2.resize(bev, (ow, oh), interpolation=cv2.INTER_AREA)
    img = cv2.imread(os.path.join(IMG_dir, "{:06d}.png".format(args.start_idx)))
    merge_img = np.hstack([bev, img])
    print(bev.shape, img.shape) # (500,300), (384, 1248)
    iminfo = merge_img.shape
    size = (iminfo[1], iminfo[0])
    print(size)

    video_path = os.path.join(args.save_dir, args.dir.split('/')[-1] + ".avi")
    print(video_path)
    videoWrite = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'MJPG'), 5, size) # fps=5

    for i in range(args.start_idx, args.end_idx):
        print("=> ", i)
        bev_path = os.path.join(BEV_dir, "{:06d}.png".format(i))
        img_path = os.path.join(IMG_dir, "{:06d}.png".format(i))
        if (not os.path.exists(bev_path)) or (not os.path.exists(img_path)):
            print("=> skip ", i)
            continue
        bev = cv2.imread(bev_path)
        bev = cv2.resize(bev, (ow, oh), interpolation=cv2.INTER_AREA)
        img = cv2.imread(img_path)
        merge_img = np.hstack([bev, img])
        # cv2.imshow("Merge View", merge_img)
        # if cv2.waitKey(0) == 27: # 'ESC'
        #     cv2.destroyAllWindows() 
        #     exit(0)
        # filename = os.path.join(args.dir, "{:06d}.png".format(i))
        # print("=> ", filename)
        # img = cv2.imread(filename)
        videoWrite.write(merge_img)

    videoWrite.release()
    print("end...")