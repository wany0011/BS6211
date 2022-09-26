import os
import cv2
import sys
import copy
import random
import pandas as pd
import numpy as np


# def list_split(_list, _train_size, _valid_size, _test_size):
#
#     _train = _list[:_train_size]
#     _valid = _list[_train_size: _train_size+_valid_size]
#     _test = _list[-_test_size:]
#     print('Train: ', len(_train), 'Valid: ', len(_valid), 'Test: ', len(_test))
#     _train.sort()
#     _valid.sort()
#     _test.sort()
#     tr_set = set(_train)
#     v_set = set(_valid)
#     te_set = set(_test)
#     if tr_set & v_set:
#         print(tr_set & v_set)
#     if te_set & v_set:
#         print(te_set & v_set)
#     if tr_set & te_set:
#         print(tr_set & te_set)
#     return _train, _valid, _test


def read_npz(ip_folder, list_, op_name):
    image_combine = None
    mask_combine = None
    coord_combine = None
    for i in range(len(list_)):
        print(ip_folder+list_[i])
        with np.load(ip_folder+list_[i]) as npz:
            print(list_[i], npz['mask'].shape, npz['coord'].shape)
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


def main():

    ip_dir = '/home/liuwei/Angio/RCA_annotated/Aug_npz/'
    op_dir = '/home/liuwei/Angio/RCA_annotated/Split{}/'.format(split_id)

    if not os.path.exists(op_dir):
        os.makedirs(op_dir)

    file_list = os.listdir(ip_dir)
    file_list.sort()
    print(file_list)

    # testing for public data
    # print('processing all...')
    # read_npz(ip_dir, file_list, op_dir + 'all')
    # sys.exit()
    # #######################

    # count frame number of each video
    video_name_list = []
    video_size_list = []
    count = None
    for file in file_list:
        v_name = file.split('_')[0]
        if v_name not in video_name_list:
            video_name_list.append(v_name)
            if count is not None:
                video_size_list.append(count)
            count = 1
        else:
            count += 1
        # print(file, count)
    video_size_list.append(count)

    # print(video_name_list)
    # print(video_size_list)

    # shuffle the list
    zip_list = list(zip(video_name_list, video_size_list))
    split_seed = 20122015 + split_id
    random.seed(split_seed)
    random.shuffle(zip_list)
    name_list, size_list = zip(*zip_list)
    name_list = list(name_list)
    size_list = list(size_list)

    for name, size in zip(name_list, size_list):
        if size != 6:
            print(name, size)

    sample_size = len(file_list)
    train_size = int(sample_size * 0.6)
    valid_size = int(sample_size * 0.2)
    test_size = int(sample_size * 0.2)
    print(train_size, valid_size, test_size)

    count = 0
    valid_name_list = []
    while count < valid_size:
        valid_name_list.append(name_list.pop(0))
        count += size_list.pop(0)

    print(valid_name_list)
    print(count)
    print(len(name_list), len(size_list))

    count = 0
    test_name_list = []
    while count < test_size:
        test_name_list.append(name_list.pop(0))
        count += size_list.pop(0)

    print(test_name_list)
    print(count)
    print(len(name_list), len(size_list))

    train_name_list = name_list

    train = []
    valid = []
    test = []
    for file in file_list:
        v_name = file.split('_')[0]
        if v_name in train_name_list:
            train.append(file)
        elif v_name in valid_name_list:
            valid.append(file)
        elif v_name in test_name_list:
            test.append(file)
        else:
            sys.exit('Unclassified file {}'.format(file))

    data = {'train': train, 'valid': valid, 'test': test}
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.transpose()
    print(df.head(5))
    print(df.tail(5))
    df.to_csv(op_dir + 'split.csv')

    print(train)
    print('processing train...')
    read_npz(ip_dir, train, op_dir + 'train_aug')
    print('processing valid...')
    read_npz(ip_dir, valid, op_dir + 'valid_aug')
    print('processing test...')
    read_npz(ip_dir, test, op_dir + 'test_aug')


if __name__ == '__main__':
    split_id = 1
    main()
