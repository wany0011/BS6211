import os
import numpy as np
import cv2

ip_npz_path = '/home/liuwei/Angio/Npz/'
video_name = 'new-4214'
op_img_path = '/home/liuwei/Angio/Npz/'

with np.load(ip_npz_path + video_name + '.npz') as npz:
    print(npz.files)
    print('loading new-{}.npz, shape {}'.format(video_name, npz['arr_0'].shape))
    # for pos in [good_idx[0], good_idx[good_idx.shape[0]//2], good_idx[-1]]:
    for pos in range(npz['arr_0'].shape[0]):
        img_name = '{}_{}.png'.format(op_img_path + video_name, pos)
        # print(npz['arr_0'][ii, 0, :, :])
        if os.path.exists(img_name):
            print('File exists.')
            continue
        cv2.imwrite(img_name, npz['arr_0'][pos, 0, :, :])
        print(img_name)
