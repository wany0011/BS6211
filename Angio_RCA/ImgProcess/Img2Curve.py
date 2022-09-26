"""
Revision history
01 - skip those with error
02 - use original size of 512*512, do NOT downsize to 128*128
"""
__UpdatedBy__ = "Patrick Wan"
__Date__ = "16-Sep-2022"
__Revision__ = "01"

"""
Script for generating curve data for input into model.

Input Dir: png files of both org img and mask 'train_good_centreline_round1'
Npz output: main output for the model. Npz contains: (1) org img, (2) mask img, (3) coordinates
Png output: for self reference
"""

import os
import cv2
import numpy as np

# from tqdm import tqdm
from ImgProcess.DrawMainCurve import MainCurve


def mask2curve(mask, op_png_name):
    """function to extract coordinates and bgr_mask (mask with crosses in BGR channels)"""

    frame = MainCurve(mask, op_png_name)

    # output origin matrix of pos
    pos_mat_origin = np.stack(frame.inds_ordered)
    # print(pos_mat_origin.shape)

    bgr_mask = frame.bgr_mask

    """  interpolate to the designated max number of points 
    # int into float
    assert mask.shape[0] == mask.shape[1], 'Input mask is not a square.'
    pos_mat_origin = pos_mat_origin / mask.shape[0]

    origin_t = np.arange(pos_mat_origin.shape[0])
    t_float = origin_t * (total_pts-1) / origin_t[-1]
    t_int = np.arange(total_pts)

    # print(origin_t)
    # print(t_float)
    # print(t_int)

    # interpolate no. of pts
    pos_mat_inter = np.zeros((total_pts, 2))
    pos_mat_inter[:, 0] = np.interp(t_int, t_float, pos_mat_origin[:, 0])
    pos_mat_inter[:, 1] = np.interp(t_int, t_float, pos_mat_origin[:, 1])

    # print(t_int)

    # print(pos_mat_origin[100: 110], t_float[100:110])
    # print(pos_mat_inter[220: 230])
    # print(pos_mat_origin[-1], pos_mat_inter[-1])

    # dis = np.linalg.norm(pos_mat_inter[1:] - pos_mat_inter[:-1], axis=1)
    # print(dis.shape, np.min(dis) * 512, np.max(dis) * 512)
    """
    return pos_mat_origin, bgr_mask


def main(total_pts):

    ip_dir = "/home/liuwei/Angio/RCA_annotated/combined_annotated/"
    op_npz_dir = "/home/liuwei/Angio/RCA_annotated/ind_npz/"
    op_png_dir = "/home/liuwei/Angio/RCA_annotated/png/"

    if not os.path.exists(op_npz_dir):
        os.makedirs(op_npz_dir)

    if not os.path.exists(op_png_dir):
        os.makedirs(op_png_dir)

    file_list = os.listdir(ip_dir)
    file_list.sort()
    print(file_list)

    for file in file_list:
        # exclude hidden files, masks and folders
        if file.startswith('.') or ('mask' in file) or (os.path.isdir(ip_dir + file)):
            continue

        id_ = ''.join(file.split('.')[0])    # for the other two
        # print(file, id_)

        # reformatted directory
        op_npz_name = op_npz_dir + id_ + '.npz'
        op_png_name = op_png_dir + id_ + '_curved.png'

        # if os.path.exists(op_npz_name):
        #     print('File exists.')
        #     continue

        mask_file_name = ip_dir + id_ + '_mask.png'
        print(mask_file_name)

        mask = cv2.imread(mask_file_name, 0)
        mask[mask < 20] = 0
        # print(np.max(mask))

        # handle no masks in folder
        if type(mask) is not np.ndarray:
            print(f'{mask_file_name} does not exist')
            continue

        #resized_mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_AREA)
        # print(mask.shape)

        # handle errors in drawing curve
        # try:
        #     coord, bgr_mask = mask2curve(resized_mask, op_png_name, total_pts=total_pts)
        # except:
        #     print(f'{id_}: issue drawing curve')
        #     continue
        # coord, bgr_mask = mask2curve(resized_mask, op_png_name, total_pts=total_pts)
        coord, bgr_mask = mask2curve(mask, op_png_name,total_pts=total_pts)

        img = cv2.imread(os.path.join(ip_dir, file), 0)

        # print('!!!', ip_dir + file, np.max(img), ip_dir, file)

        #resized_img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)

        # print(img.shape, type(img))
        # label renamed as coord in augmented data

        # save npz file
        #np.savez_compressed(op_npz_name, img=resized_img, mask=resized_mask, coord=coord)
        np.savez_compressed(op_npz_name, img=img, mask=mask, coord=coord)

        # save png file format: (org image) concat (img overlay coords)
        combined = cv2.hconcat([np.stack((resized_img,)*3, axis=-1), bgr_mask])
        cv2.imwrite(op_png_name, combined)


if __name__ == '__main__':
    main(total_pts=256)     # standardize all curves to 256 pts
