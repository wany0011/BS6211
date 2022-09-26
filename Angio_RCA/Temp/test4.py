import os
import cv2
import pandas as pd
import numpy as np


ip_npz_path = '/home/liuwei/Angio/Npz/video/'

# op_img_path = '/home/liuwei/Angio/Image/LAO_Straight/'

npz_list = os.listdir(ip_npz_path)
npz_list.sort()
print(npz_list)

for video_name in npz_list:
    with np.load(ip_npz_path + video_name) as npz:
        video_id = video_name[4:-4]
        print(npz.files)
        print('loading new-{}.npz, shape {}'.format(video_name, npz['arr_0'].shape))
        # for pos in [good_idx[0], good_idx[good_idx.shape[0]//2], good_idx[-1]]:
        for pos in range(len(npz['arr_0'])):
            img_name = '{}_{}.png'.format(ip_npz_path+video_name, pos)
            # print(npz['arr_0'][ii, 0, :, :])
            if os.path.exists(img_name):
                print('File exists.')
                continue
            cv2.imwrite(img_name, npz['arr_0'][pos, 0, :, :])
            print(img_name)
