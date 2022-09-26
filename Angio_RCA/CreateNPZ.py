"""
Revision history
01 - 1st version
"""
__CreatedBy__ = "Patrick Wan"
__Date__ = "16-Sep-2022"
__Revision__ = "01"

import json
import numpy as np
import os
import cv2
from ImgProcess.Img2Curve import mask2curve
from Utils.Visualizer import draw_pts
from ImgProcess.Augmentation import augmentation
from PIL import Image

with open("./Config.json", "r") as f:
    print(os.getcwdb())
    config = json.load(f)


def create_npz(augment_rounds, final_size):

    ip_dir = config["image_proc"]["ip_dir"]
    op_dir = config["image_proc"]["op_dir"]
    op_npz_dir = config["image_proc"]["op_npz_dir"]
    op_png_dir = config["image_proc"]["op_png_dir"]

    file_list = os.listdir(ip_dir)
    file_list.sort()
    print(file_list)

    dict_all = {}
    loop = 0
    for file in file_list:
        if file.startswith('.') or 'mask' in file:
            continue
        prefix = file.split('.')[0]

        # read mask file
        mask_file_name = ip_dir + prefix + '_mask.png'
        # if not os.path.exists(mask_file_name):
        #    continue
        # print(mask_file_name)

        mask = cv2.imread(mask_file_name, 0)
        # print(mask.shape, np.max(mask), np.min(mask))
        mask[mask >= 30] = 255
        mask[mask < 20] = 0

        op_npz_name = op_npz_dir + prefix + '.npz'
        op_png_name = op_png_dir + prefix + '_curved.png'

        # resized_mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_AREA)
        try:
            coord, bgr_mask = mask2curve(mask, op_png_name)
            # print(coord.shape)
        except ValueError as e:
            continue

        img = cv2.imread('{}{}.png'.format(ip_dir, prefix), 0)
        # np.savez_compressed(op_npz_name, img=img, mask=mask, coord=coord)

        # if np.random.randint(100) < 20:
        # bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # bgr_mask = np.max(bgr_mask) - bgr_mask
        # labeled_img = draw_pts(bgr_img, coord, n_pts=32)
        # cv2.imwrite(op_png_name, np.concatenate((bgr_img, bgr_mask, labeled_img), axis=1))

        lst_id = prefix.split('_')
        if lst_id[0] not in dict_all:
            dict_all[lst_id[0]] = list()

        dict_individual = {str(0): (img, mask, coord)}

        # augment data
        img = Image.open(ip_dir + file).convert('L')
        mask = Image.open(mask_file_name).convert('L')

        for i in range(augment_rounds):
            trans_img, trans_mask = augmentation(img, mask, final_size)
            try:
                op_png_name = '{}{}_{}_curved.png'.format(op_png_dir, prefix, i+1)
                coord, bgr_mask = mask2curve(trans_mask, op_png_name)
            except ValueError as e:
                cv2.imwrite(op_png_name, e)
                continue
            dict_individual[str(i+1)] = (trans_img, trans_mask, coord)

        dict_all[lst_id[0]].append((file, dict_individual))
        loop += 1
        print("{}_ file={}: len={}".format(loop, file, len(dict_individual)))

    np.savez_compressed(op_dir+'dump', **dict_all)

    """
    dict_test = {}
    dict_test = np.load(op_dir+'dump.npz', allow_pickle=True)
    for key, val in sorted(dict_test.items()):
        # ID
        print(key)
        for i in range(val.shape[0]):
            # file-name that belongs to the same ID
            print(val[i, 0])
            for idx in val[i, 1].keys():
                # 0- original, augmented from 1 to augment_rounds
                print(idx)
                # image
                print(val[0, 1][idx][0].shape)
                # mask
                print(val[0, 1][idx][1].shape)
                # points
                print(val[0, 1][idx][2].shape)
    """


if __name__ == '__main__':
    create_npz(augment_rounds=5,  final_size=512)

