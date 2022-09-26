import os
import cv2
import sys
import copy
import random
import pandas as pd
import numpy as np
# from tqdm import tqdm


def read_npz(list_, op_name):
    image_combine = None
    mask_combine = None
    coord_combine = None
    id_combine = None
    for i in range(len(list_)):
        # print(list_[i])
        id_ = list_[i].split('/')[-1].split('.')[0].split('_')
        video = int(id_[0])
        frame = int(id_[1])
        if len(id_) > 2:
            aug = int(list_[i].split('/')[-1].split('.')[0].split('_')[2])
        else:
            aug = 0
        with np.load(list_[i]) as npz:
            # print(list_[i], id_, np.min(npz['img']), np.max(npz['img']))
            # print(npz['img'].shape)
            # continue
            if i == 0:
                image_combine = npz['img'][np.newaxis, :]
                mask_combine = npz['mask'][np.newaxis, :]
                coord_combine = npz['coord'][np.newaxis, :]
                id_combine = np.asarray([video, frame, aug])[np.newaxis, :]
            else:
                image_combine = np.concatenate((image_combine, npz['img'][np.newaxis, :]), axis=0)
                mask_combine = np.concatenate((mask_combine, npz['mask'][np.newaxis, :]), axis=0)
                coord_combine = np.concatenate((coord_combine, npz['coord'][np.newaxis, :]), axis=0)
                id_combine = np.concatenate((id_combine, np.asarray([video, frame, aug])[np.newaxis, :]), axis=0)
            # cv2.imshow('ImageWindow', npz[-1])
            # cv2.waitKey()
    img_shape = image_combine.shape
    mask_shape = mask_combine.shape
    coord_shape = coord_combine.shape
    id_shape = id_combine.shape
    print(image_combine.shape, mask_combine.shape, coord_combine.shape, id_combine.shape)
    # print(id_combine, 'reached here.')
    # np.savez(op_name, img=image_combine, mask=mask_combine, coord=coord_combine, id=id_combine)


def read_npz_wo_new_axis(list_, op_name):
    image_combine = None
    mask_combine = None
    coord_combine = None
    id_combine = None
    for i in range(len(list_)):
        print(list_[i])
        with np.load(list_[i]) as npz:
            print(list_[i], npz['coord'].shape)
            if i == 0:
                image_combine = npz['img']
                # mask_combine = npz['mask']
                coord_combine = npz['coord']
                # id_combine = npz['id']
            else:
                image_combine = np.concatenate((image_combine, npz['img']), axis=0)
                # mask_combine = np.concatenate((mask_combine, npz['mask']), axis=0)
                coord_combine = np.concatenate((coord_combine, npz['coord']), axis=0)
                # id_combine = np.concatenate((id_combine, npz['id']), axis=0)
            # cv2.imshow('ImageWindow', npz[-1])
            # cv2.waitKey()

    # img_shape = image_combine.shape
    print(image_combine.shape,
          coord_combine.shape,
          id_combine.shape
          )
    coord_combine = coord_combine.reshape((-1, 256, 2))
    np.savez(op_name,
             img=image_combine,
             coord=coord_combine,
             id=id_combine
            )


def main():

    # ip_dir = '/home/liuwei/Angio/Npz/MainCurve_LAO/Individual/New_test/'
    # op_dir = '/home/liuwei/Angio/Npz/MainCurve_LAO/FixedSplit/'
    #
    # if not os.path.exists(op_dir):
    #     os.makedirs(op_dir)
    #
    # file_list = os.listdir(ip_dir)
    # file_list.sort()
    # print(file_list)
    #
    # read_npz(ip_dir, file_list, op_dir + 'test')

    # op_dir = '/home/liuwei/Angio/Npz/MainCurve_LAO/FixedSplit/'
    #
    # if not os.path.exists(op_dir):
    #     os.makedirs(op_dir)
    #
    # ip_dir = '/home/liuwei/Angio/Npz/MainCurve_LAO/Individual/Old_train/'
    # file_list = os.listdir(ip_dir)
    # all_in_one = [ip_dir + i for i in file_list]
    # print(len(all_in_one))
    #
    # ip_dir = '/home/liuwei/Angio/Npz/MainCurve_LAO/Individual/New_train/'
    # file_list = os.listdir(ip_dir)
    # all_in_one += [ip_dir + i for i in file_list]
    # print(len(all_in_one))
    #
    # ip_dir = '/home/liuwei/Angio/Npz/MainCurve_LAO/Individual/New_train2/'
    # file_list = os.listdir(ip_dir)
    # all_in_one += [ip_dir + i for i in file_list]
    # print(len(all_in_one))
    #
    # all_in_one.sort()
    #
    # size = len(all_in_one) // 5
    #
    # for blk in range(5):
    #     # print(size*blk, min(size*(blk+1), len(all_in_one)), min(size*(blk+1), len(all_in_one)) - size*blk)
    #     if blk == 4:
    #         read_npz(all_in_one[size*blk:], '{}train_blk{}'.format(op_dir, blk))
    #     else:
    #         read_npz(all_in_one[size * blk: size * (blk + 1)], '{}train_blk{}'.format(op_dir, blk))

    ip_dir = '/home/chentyt/Documents/4tb/Tiana/P100/Data/RCA_annotated/v2_patch_npz/'
    op_dir = '/home/chentyt/Documents/4tb/Tiana/P100/Data/RCA_Split1/'

    list_ = ['train_aug.npz', 'train_ind.npz', 'train_Tiana.npz']

    list_ = os.listdir(ip_dir)

    # list of full directories of all files
    all_in_one = [ip_dir + i for i in list_]
    read_npz_wo_new_axis(all_in_one, op_dir + 'v2_test_patch.npz')  # change the output name


if __name__ == '__main__':
    split_id = 0
    main()
