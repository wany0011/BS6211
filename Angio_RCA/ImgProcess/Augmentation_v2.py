import os
import cv2
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

from Img2Curve import mask2curve
from tqdm import tqdm

# r_a: rotate angle range
# s_a: shear angle range
r_a = 15
s_a = 10


def augmentation(img, mask, final_size):
    # randomly generating images from a random combination of rotations and shearing
    rotate_angle = np.random.uniform(-r_a, r_a)
    shear_angle = np.random.uniform(-s_a, s_a)
    mask_array = np.array(mask)
    # print(np.max(mask_array))
    row_center, col_center = np.argwhere(mask_array == 255).sum(0)/((mask_array == 255).sum())
    # print(row_center, col_center)
    img = transforms.functional.affine(img, angle=rotate_angle,
                                       translate=[0, 0],
                                       scale=1, shear=shear_angle)
    mask = transforms.functional.affine(mask, angle=rotate_angle,
                                        translate=[0, 0],
                                        scale=1, shear=shear_angle)

    mask_array = np.array(mask)
    pos = np.argwhere(mask_array == 255)
    # print(pos)
    up_bound, down_bound = np.min(pos[:, 0]), np.max(pos[:, 0])
    left_bound, right_bound = np.min(pos[:, 1]), np.max(pos[:, 1])

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

    mask_array = mask_array[real_up: real_down, real_left: real_right]

    img_array = np.array(img)
    img_array = img_array[real_up: real_down, real_left: real_right]
    img_array[img_array == 0] = np.mean(img_array[img_array > 50])

    resized_mask = cv2.resize(mask_array, (final_size, final_size), interpolation=cv2.INTER_AREA)
    resized_img = cv2.resize(img_array, (final_size, final_size), interpolation=cv2.INTER_AREA)

    return resized_img, resized_mask


def main(augment_rounds, final_size, total_pts):
    """ Input: png curve image (org image, mask). """

    ip_dir = "/home/chentyt/Documents/4tb/Tiana/Centreline_annotation/LAO_Straight/round1/train_pseudoannotations/png/"
    op_npz_dir = "/home/chentyt/Documents/4tb/Tiana/Centreline_annotation/LAO_Straight/round1/train_pseudoannotations/augmentation/npz/"
    op_png_dir = "/home/chentyt/Documents/4tb/Tiana/Centreline_annotation/LAO_Straight/round1/train_pseudoannotations/augmentation/png/"

    if not os.path.exists(op_npz_dir):
        os.makedirs(op_npz_dir)

    if not os.path.exists(op_png_dir):
        os.makedirs(op_png_dir)

    file_list = os.listdir(ip_dir)
    file_list.sort()
    print(file_list)

    for file in tqdm(file_list):

        if not os.path.exists(ip_dir + file):
            continue

        if file.startswith('.') or 'curved' not in file:
            continue

        id_ = '_'.join(file.split('_')[:2])  # for the other two
        # print(file, id_)
        op_npz_name = op_npz_dir + id_ + 'aug.npz'
        op_png_name = op_png_dir + id_ + '_aug_new.png'

        # if os.path.exists(op_npz_name):
        #     print('File exists.')
        #     continue
        combine_img = cv2.imread(ip_dir + id_ + '_curved.png', 0)

        # print(combine_img.shape, np.max(combine_img))
        #
        # plt.figure()
        # plt.imshow(combine_img, cmap='gray')
        # plt.show()

        #open img and mask, convert to greyscale
        img = Image.fromarray(combine_img[:, :128]).convert('L')
        mask = Image.fromarray(combine_img[:, -128:]).convert('L')
        # print(np.max(img), np.max(mask))
        # mask.show()

        for i in range(augment_rounds):

            op_npz_name = '{}{}_{}.npz'.format(op_npz_dir, id_, i)
            op_png_name = '{}{}_{}_curved.png'.format(op_png_dir, id_, i)

            # print('Generating...', op_npz_name)

            if os.path.exists(op_npz_name):
                # print('File exists.', op_npz_name)
                continue

            trans_img, trans_mask = augmentation(img, mask, final_size)

            assert trans_img.shape == (final_size, final_size), 'trans_img shape {}'.format(trans_img.shape)

            # original version ####
            try:
                coord, bgr_mask = mask2curve(trans_mask, op_png_name, total_pts=total_pts)
                np.savez_compressed(op_npz_name, img=trans_img, mask=trans_mask, coord=coord)
            except:
                print('error generating {} in augment round {}.'.format(file, i))

            if np.random.randint(100) < 20: # sample only 20% for checking manually, else, file is too big
                # bgr_img = cv2.cvtColor(trans_img, cv2.COLOR_GRAY2BGR)
                bgr_img = np.stack((trans_img,)*3, axis=-1)
                # bgr_mask = np.max(bgr_mask) - bgr_mask
                op_img = np.concatenate((bgr_img, bgr_mask), axis=1)
                op_img = cv2.resize(op_img, (1024, 512), interpolation=cv2.INTER_AREA)
                cv2.imwrite(op_png_name, op_img)

            # for pix2pix ####
            # cv2.imwrite(op_png_name, np.concatenate((trans_img, trans_mask), axis=1))


if __name__ == '__main__':
    main(augment_rounds=10, final_size=128, total_pts=256)
