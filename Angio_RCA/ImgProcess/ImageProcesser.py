"""
Revision history
01 - skip those with error
02 - use original size of 512*512, do NOT downsize to 128*128
03 - do not limit the sampling point of center-line to fixed &
        pre-configured "total_pts", make it a variable length instead
"""
__UpdatedBy__ = "Patrick Wan"
__Date__ = "16-Sep-2022"
__Revision__ = "03"


import os

import cv2
import numpy as np

from ImgProcess.Img2Curve import mask2curve

'''automate find image files in a folder which are manually draw the curve. Move them to another directory, generate
concatenated image and npz'''
import json
from Utils.Visualizer import draw_pts

print("Current working directory: {0}".format(os.getcwd()))
with open("./Config.json", "r") as f:
    # Print the current working directory
    print("Current working directory: {0}".format(os.getcwd()))
    config = json.load(f)


def main():
    # ip_dir = "/home/liuwei/Angio/RCA_annotated/combined_annotated/"
    # op_npz_dir = "/home/liuwei/Angio/RCA_annotated/ind_npz/"
    # op_png_dir = "/home/liuwei/Angio/RCA_annotated/png/"

    ip_dir = config["image_proc"]["ip_dir"]
    # op_dir = config["image_proc"]["op_dir"]
    op_npz_dir = config["image_proc"]["op_npz_dir"]
    op_png_dir = config["image_proc"]["op_png_dir"]

    if not os.path.exists(op_npz_dir):
        os.makedirs(op_npz_dir)

    if not os.path.exists(op_png_dir):
        os.makedirs(op_png_dir)

    file_list = os.listdir(ip_dir)
    file_list.sort()
    print(file_list)

    for file in file_list:

        if not os.path.exists(ip_dir+file):
            continue

        if file.startswith('.') or 'mask' not in file:
            continue

        id_ = '_'.join(file.split('_')[:3])    # for the other two
        # print(file, id_)
        op_npz_name = op_npz_dir + id_ + '.npz'
        op_png_name = op_png_dir + id_ + '_curved.png'

        # if os.path.exists(op_npz_name):
        #     print('File exists.')
        #     continue

        mask_file_name = ip_dir + id_ + '_mask.png'
        if not os.path.exists(mask_file_name):
            continue

        # print(mask_file_name)
        mask = cv2.imread(mask_file_name, cv2.IMREAD_GRAYSCALE)
        # print(mask.shape, np.max(mask), np.min(mask))
        mask[mask >= 30] = 255
        mask[mask < 20] = 0

        # resized_mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_AREA)
        try:
            coord, bgr_mask = mask2curve(mask, op_png_name)
            # print(coord.shape)
        except ValueError as e:
            continue

        img = cv2.imread('{}{}.png'.format(ip_dir, id_), cv2.IMREAD_GRAYSCALE)
        print('!!!', ip_dir+file, np.max(img), ip_dir, file)
        # resized_img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

        # print(img.shape, type(img))
        # label renamed as coord in augmented data
        # np.savez_compressed(op_npz_name, img=resized_img, mask=resized_mask, coord=coord)
        np.savez_compressed(op_npz_name, img=img, mask=mask, coord=coord)

        print(id_, np.max(img), np.min(img))
        # bgr_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2BGR)
        # msk_img = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
        bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        msk_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        labeled_img = draw_pts(bgr_img, coord, n_pts=32)
        combine_img = np.concatenate((bgr_img, msk_img, labeled_img), axis=1)

        # combine_img = cv2.resize(combine_img, (512*3, 512), interpolation=cv2.INTER_AREA)
        # print(bgr_img.shape)
        cv2.imwrite(op_png_name, combine_img)
        # command = 'rm {}{}*'.format(ip_dir, id_)
        # print(command)
        # os.system(command)
        # sys.exit()


if __name__ == '__main__':
    main()
