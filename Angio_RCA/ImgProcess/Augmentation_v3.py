import torch
import numpy as np
import cv2
import os

import Config
from Dataset.DataSet_old import DataSet

ip_dir = Config.ip_dir
op_dir = Config.op_dir
seed = Config.seed
epoch_no = Config.epoch_no
patience = Config.patience
lr = Config.lr

model_id = Config.model_id
n_channel_2d = Config.n_channel_2d
n_downsampling = Config.n_downsampling
out_kernel = Config.out_kernel
in_kernel = Config.in_kernel
n_feature = Config.n_feature
n_channel_1d = Config.n_channel_1d
neuron_list = Config.neuron_list
arch = Config.arch

batch_size = Config.batch_size
num_pts = Config.num_pts
coord_loss_w = Config.coord_loss_w
random_pts = Config.random_pts
use_aug = Config.use_aug
rotation_range = Config.rotation_range
shear_range = Config.shear_range
crop_range = Config.crop_range

encoder = Config.encoder
decoder = Config.decoder
feature_net = Config.feature_net

load_weight = Config.load_weight


def draw_pts(img_, xy, points):

    for pt in range(points):

        row = xy[pt, 0]
        col = xy[pt, 1]

        img_[row-3:row+4, col, :] = [255, 0, 0]
        img_[row, col-3:col+4, :] = [255, 0, 0]

    return img_


def main(augment_rounds, final_size, total_pts):
    """ augment by using on-the-fly function in dataset. """

    if not os.path.exists(op_npz_dir):
        os.makedirs(op_npz_dir)

    if not os.path.exists(op_png_dir):
        os.makedirs(op_png_dir)

    for npz_name in ip_list:

        torch.cuda.empty_cache()

        img_combine = None
        coord_combine = None
        id_combine = None
        start_flag = True

        data_set = DataSet(ip_dir, npz_name, 256, random_pts=False,
                           aug=[rotation_range, shear_range, crop_range], jit=True, get_id=True)

        for i in range(data_set.len()):
            # img, t, coord, id_ = data_set[i]
            # print(id_, data_set.len())
            for j in range(augment_rounds):
                img, t, coord, id_ = data_set[i]
                # print(img.shape, coord.shape, np.max(img), np.max(coord))
                # print(t)

                if id_[0] in [1008, 1021, 1080, 1216, 1237]:
                    continue

                if start_flag:
                    img_combine = img * 255
                    coord_combine = coord[np.newaxis, :]
                    id_combine = id_[np.newaxis, :]
                    start_flag = False
                else:
                    img_combine = np.concatenate((img_combine, img * 255), axis=0)
                    coord_combine = np.concatenate((coord_combine, coord[np.newaxis, :]), axis=0)
                    id_combine = np.concatenate((id_combine, id_[np.newaxis, :]), axis=0)

                if save_pic:
                    if np.random.rand() < 0.01:
                    # if j == 0:
                        img = cv2.cvtColor(np.uint8(img[0, ...] * 255), cv2.COLOR_GRAY2BGR)
                        img1 = draw_pts(np.copy(img), (128 * coord).astype(int), points=256)
                        print(img.shape, img1.shape)
                        img_all = np.concatenate((img, img1), axis=1)
                        cv2.imwrite('{}{}_{}_{}.png'.format(op_png_dir, id_[0], id_[1], j), img_all)

            print(i, img_combine.shape, coord_combine.shape, id_combine.shape)
        np.savez('{}trail{}_aug_{}'.format(op_npz_dir, augment_rounds, npz_name), img=img_combine, coord=coord_combine, id=id_combine)


if __name__ == '__main__':
    ip_dir = '/home/liuwei/Angio/Npz/MainCurve_LAO/FixedSplit/'
    op_npz_dir = '/home/liuwei/Angio/Npz/MainCurve_LAO/FixedSplit/'
    op_png_dir = '/home/liuwei/Angio/Image/LAO_Main_Curved/Test_Aug/'
    # ip_list = ['train_wo_blk0']
    ip_list = ['train_blk1', 'train_blk2', 'train_blk3', 'train_blk4']
    save_pic = True
    main(augment_rounds=10, final_size=128, total_pts=256)
