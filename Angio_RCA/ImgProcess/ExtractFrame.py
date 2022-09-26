import os
import cv2
import pandas as pd
import numpy as np

split_file = '/home/liuwei/Angio/train_test_valid_split.csv'

npz_csv_path = '/home/liuwei/Angio/LAO-csv/new-csv/'

ip_npz_path = '/home/liuwei/local_mnt/Angio/data_26-8-2020/NHC Processed Data/'

op_img_path = '/home/liuwei/Angio/Image/LAO_Straight/'

split_csv = pd.read_csv(split_file)

train_list = []
valid_list = []
test_list = []

for index, line in split_csv.iterrows():
    if line[2] == 'train':
        train_list.append(line[1])
    elif line[2] == 'valid':
        valid_list.append(line[1])
    elif line[2] == 'test':
        test_list.append(line[1])
    else:
        print('Unknown class.', line)

print(train_list)
print(valid_list)
print(test_list)

# csv_list = os.listdir(npz_csv_path)
# csv_list.sort()
# print(csv_list)

for csv_name in sorted(os.listdir(npz_csv_path)):
    if not csv_name.startswith('new'):
        continue

    video_name = csv_name.split('.')[0].split('-')[-1]
    print(csv_name, video_name)

    if int(video_name) in train_list:
        op_img_path_ = op_img_path + 'train/'
    elif int(video_name) in valid_list:
        op_img_path_ = op_img_path + 'valid/'
    elif int(video_name) in test_list:
        op_img_path_ = op_img_path + 'test/'
    else:
        print('Unknown class. ', video_name)

    csv_file = pd.read_csv(npz_csv_path + csv_name)
    idx = csv_file['0'].dropna().to_numpy()
    label = csv_file['1'].dropna().to_numpy()
    # print(idx, idx.shape)
    # print(label, label.shape)
    assert idx.shape == label.shape, 'idx shape {} is not identical with label shape {}'.format(idx.shape, label.shape)
    good_idx = idx[label == 0]
    good_range = good_idx.shape
    print(good_idx, good_range)
    # print(good_idx[0], good_idx[good_idx.shape[0]//2], good_idx[-1])

    gap = 3        # gap is how many frames between the extracted frames
    choose_list = np.asarray(range(good_idx[0]+1, good_idx[-1], gap))
    print(gap, choose_list)

    # print('loading {}new-{}.npz'.format(op_img_path_, video_name))
    with np.load(ip_npz_path + video_name + '.npz') as npz:
        print(npz.files)
        print('loading new-{}.npz, shape {}'.format(video_name, npz['arr_0'].shape))

        for pos in choose_list:
            img_name = '{}_{}.png'.format(op_img_path_ + video_name, pos)
            good_npz = npz['arr_0'][pos]

            id_ = np.zeros((len(good_idx), 2), dtype=int)
            id_[:, 0] = video_name
            id_[:, 1] = good_idx
            # print(id_)
            cv2.imwrite(img_name, good_npz)
