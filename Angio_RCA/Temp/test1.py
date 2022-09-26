import numpy as np
import cv2

ip_npz_path = '/home/liuwei/Angio/Npz/RAOCaudal/'

with np.load(ip_npz_path + 'new-1377.npz') as npz:
    print(npz.files)
    print('loading new-{}.npz, shape {}'.format(10, npz['arr_0'].shape))

    good_npz = npz['arr_0']
    for pos in range(len(good_npz)):
        img_name = '{}_{}.png'.format(ip_npz_path + '10', pos)
        # good_npz = npz['arr_0'][pos]

        # id_ = np.zeros((len(good_idx), 2), dtype=int)
        # id_[:, 0] = video_name
        # id_[:, 1] = good_idx
        # print(id_)
        cv2.imwrite(img_name, good_npz[pos, 0, :, :])
