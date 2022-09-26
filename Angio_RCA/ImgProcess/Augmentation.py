"""
Revision history
01 - skip those with error
02 - use original size of 512*512, do NOT downsize to 128*128
"""
__UpdatedBy__ = "Patrick Wan"
__Date__ = "16-Sep-2022"
__Revision__ = "02"

import os
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np

from ImgProcess.Img2Curve import mask2curve

r_a = 15
s_a = 10

import Config
from Utils.Visualizer import draw_pts


def augmentation(img, mask, final_size):
    rotate_angle = np.random.uniform(-r_a, r_a)
    shear_angle = np.random.uniform(-s_a, s_a)
    mask_array = np.array(mask)
    # print(np.max(mask_array))
    row_center, col_center = np.argwhere(mask_array > 20).sum(0)/((mask_array > 20).sum())
    # print(row_center, col_center, (mask_array > 20).sum(), np.max(mask_array))
    img = transforms.functional.affine(img, angle=0,
                                       translate=[row_center/512, col_center/512],
                                       scale=1, shear=0)
    img = transforms.functional.affine(img, angle=rotate_angle,
                                       translate=[0, 0],
                                       scale=1, shear=shear_angle)

    mask = transforms.functional.affine(mask, angle=0,
                                        translate=[row_center/512, col_center/512],
                                        scale=1, shear=0)
    mask = transforms.functional.affine(mask, angle=rotate_angle,
                                        translate=[0, 0],
                                        scale=1, shear=shear_angle)

    mask_array = np.array(mask)
    pos = np.argwhere(mask_array > 20)
    # print(mask_array.shape, pos.shape, np.max(mask_array))
    up_bound, down_bound = np.min(pos[:, 0]) - 5, np.max(pos[:, 0]) + 5
    left_bound, right_bound = np.min(pos[:, 1]) - 5, np.max(pos[:, 1]) + 5
    # print(mask_array.shape, pos.shape, np.max(mask_array))

    try:
        real_up = np.random.randint(0, up_bound - 30)
    except ValueError:
        real_up = 0

    try:
        real_down = np.random.randint(down_bound + 30, mask_array.shape[0])
    except ValueError:
        real_down = mask_array.shape[0]

    try:
        real_left = np.random.randint(0, left_bound - 30)
    except ValueError:
        real_left = 0

    try:
        real_right = np. random.randint(right_bound + 30, mask_array.shape[1])
    except ValueError:
        real_right = mask_array.shape[1]

    #print(up_bound, real_up, down_bound, real_down, left_bound, real_left, right_bound, real_right)

    mask_array = mask_array[real_up: real_down, real_left: real_right]

    img_array = np.array(img)
    img_array = img_array[real_up: real_down, real_left: real_right]
    img_array[img_array < 10] = np.mean(img_array[img_array > 10])

    resized_mask = cv2.resize(mask_array, (final_size, final_size), interpolation=cv2.INTER_AREA)
    resized_img = cv2.resize(img_array, (final_size, final_size), interpolation=cv2.INTER_AREA)

    return resized_img, resized_mask


def main(augment_rounds, final_size):
    """ Reading raw image and the mask image separately. """
    #ip_dir = '/home/liuwei/Angio/RCA_annotated/combined_annotated/'
    #op_npz_dir = '/home/liuwei/Angio/RCA_annotated/Aug_npz/'
    #op_png_dir = '/home/liuwei/Angio/RCA_annotated/Aug_png/'
    ip_dir = Config.ip_dir
    op_npz_dir = Config.op_npz_dir
    op_png_dir = Config.op_png_dir


    if not os.path.exists(op_npz_dir):
        os.makedirs(op_npz_dir)

    if not os.path.exists(op_png_dir):
        os.makedirs(op_png_dir)

    file_list = os.listdir(ip_dir)
    file_list.sort()
    print(file_list)

    for file in file_list:
        if file.startswith('.') or 'mask' in file:
            continue

        id_ = file.split('.')[0]

        mask_file_name = ip_dir + id_ + '_mask.png'
        print(id_, file, mask_file_name)

        img = Image.open(ip_dir + file).convert('L')
        mask = Image.open(mask_file_name).convert('L')

        for i in range(augment_rounds):

            op_npz_name = '{}{}_{}.npz'.format(op_npz_dir, id_, i)
            op_png_name = '{}{}_{}_curved.png'.format(op_png_dir, id_, i)

            if os.path.exists(op_npz_name):
                print('File exists.', op_npz_name)
                continue

            trans_img, trans_mask = augmentation(img, mask, final_size)

            assert trans_img.shape == (final_size, final_size), 'trans_img shape {}'.format(trans_img.shape)

            # original version ####
            try:
                coord, bgr_mask = mask2curve(trans_mask, op_png_name)
            except ValueError as e:
                cv2.imwrite(op_png_name, e)
                continue

            np.savez_compressed(op_npz_name, img=trans_img, mask=trans_mask, coord=coord)

            #if np.random.randint(100) < 20:
            bgr_img = cv2.cvtColor(trans_img, cv2.COLOR_GRAY2BGR)
            bgr_mask = np.max(bgr_mask) - bgr_mask
            labeled_img = draw_pts(bgr_img, coord, n_pts=32)
            cv2.imwrite(op_png_name, np.concatenate((bgr_img,bgr_mask,labeled_img), axis=1))

            # for pix2pix ####
            # cv2.imwrite(op_png_name, np.concatenate((trans_img, trans_mask), axis=1))


if __name__ == '__main__':
    main(augment_rounds=5, final_size=512)
