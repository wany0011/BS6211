import os
import cv2
import sys
import copy
import random
import pandas as pd
import numpy as np


def read_npz(ip_folder, list_, op_name):
    image_combine = None
    mask_combine = None
    coord_combine = None
    for i in range(len(list_)):
        with np.load(ip_folder+list_[i]) as npz:
            print(list_[i])
            if i == 0:
                image_combine = npz['img'][np.newaxis, :]
                mask_combine = npz['mask'][np.newaxis, :]
                coord_combine = npz['coord'][np.newaxis, :]
            else:
                image_combine = np.concatenate((image_combine, npz['img'][np.newaxis, :]), axis=0)
                mask_combine = np.concatenate((mask_combine, npz['mask'][np.newaxis, :]), axis=0)
                coord_combine = np.concatenate((coord_combine, npz['coord'][np.newaxis, :]), axis=0)
            # cv2.imshow('ImageWindow', npz[-1])
            # cv2.waitKey()
    print(image_combine.shape, mask_combine.shape, coord_combine.shape)
    np.savez(op_name, img=image_combine, mask=mask_combine, coord=coord_combine)


def main(split_id):

    ip_dir = '/home/liuwei/Angio/Curve/MainCurve_LAO/Individual/Aug/'
    op_dir = '/home/liuwei/Angio/Curve/MainCurve_LAO/Split{}/'.format(split_id)

    if not os.path.exists(op_dir):
        os.makedirs(op_dir)

    file_list = os.listdir(ip_dir)
    file_list.sort()
    # print(file_list)

    csv_name = op_dir + 'split.csv'
    csv = pd.read_csv(csv_name)

    train_name_list = [x[:-4] for x in csv['train'].dropna().to_list()]
    valid_name_list = [x[:-4] for x in csv['valid'].dropna().to_list()]
    test_name_list = [x[:-4] for x in csv['test'].dropna().to_list()]

    print('train name', len(train_name_list), train_name_list)
    print('valid name', len(valid_name_list), valid_name_list)
    print('test name', len(test_name_list), test_name_list)

    train = []
    valid = []
    test = []
    for file in file_list:
        v_name = '_'.join(file.split('_')[:2])
        print(v_name)
        if v_name in train_name_list:
            train.append(file)
        elif v_name in valid_name_list:
            valid.append(file)
        elif v_name in test_name_list:
            test.append(file)
        else:
            sys.exit('Unclassified file {}'.format(file))

    print('train', len(train))
    print('valid', len(valid))
    print('test', len(test))

    print('processing train...')
    read_npz(ip_dir, train, op_dir + 'train_aug_50')
    # print('processing valid...')
    # read_npz(ip_dir, valid, op_dir + 'valid_aug')
    # print('processing test...')
    # read_npz(ip_dir, test, op_dir + 'test_aug')


if __name__ == '__main__':
    main(split_id=0)
