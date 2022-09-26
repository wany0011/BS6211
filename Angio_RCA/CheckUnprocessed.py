import torch
import numpy as np
import sys
import cv2
import os

import Config
# import resource
from Device import mydevice
from SystemLogs import SystemLogs
from Networks import FinalModel
# from SupervisedTrainer import SupervisedTrainer
# from model_IO import training_model

""" Script for predicting centrelines from pretrained model"""

# load parameters
model_dir = Config.op_dir
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
# num_pts = Config.num_pts
num_pts = 256
coord_loss_w = Config.coord_loss_w
random_pts = Config.random_pts
use_aug = Config.use_aug
rotation_range = Config.rotation_range
shear_range = Config.shear_range
crop_range = Config.crop_range


encoder = Config.encoder
decoder = Config.decoder
feature_net = Config.feature_net


def draw_pts(img, pred_xy, op_name):
    """
    Function to overlay original with predicted centreline.
    """
    # original img
    bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # cv2.imwrite('test.png', img)
    # bgr_img = 255 - bgr_img
    overlap = np.copy(bgr_img)
    blank_img = np.zeros(bgr_img.shape, dtype=np.uint8)

    for pt in range(num_pts):

        row = int(pred_xy[pt, 0] * img.shape[0])
        col = int(pred_xy[pt, 1] * img.shape[1])

        # overlap img with predictions
        overlap[row-3:row+4, col, :] = [0, 0, 255]
        overlap[row, col-3:col+4, :] = [0, 0, 255]

        # just the centreline img
        blank_img[row-3:row+4, col, :] = 255
        blank_img[row, col-3:col+4, :] = 255

    combine_img = np.concatenate((bgr_img, overlap, blank_img), axis=1)
    # print(bgr_img.shape)
    cv2.imwrite(op_name, combine_img)
    return


def checker():
    torch.cuda.empty_cache()
    SystemLogs(mydevice)  # print the hostname, pid, device etc

    net = FinalModel(encoder=encoder,
                     decoder=decoder,
                     feature_net=feature_net,
                     n_channel_2d=n_channel_2d,
                     n_downsampling=n_downsampling,
                     out_kernel=out_kernel,
                     in_kernel=in_kernel,
                     n_feature=n_feature,
                     n_channel_1d=n_channel_1d,
                     neuron_list=neuron_list,
                     n_pts=num_pts)

    weight_path = '{}{}_{}_weight'.format(model_dir, model_id, arch)
    print('Loading: ', weight_path)
    if not os.path.exists(weight_path):
        sys.exit('No trained model exists.')

    checkpoint = torch.load(weight_path)
    net.load_state_dict(checkpoint['net_state_dict'])
    net.to(mydevice)
    net.eval()
    epoch = checkpoint['epoch']
    print('Epoch:', epoch)

    file_list = os.listdir(ip_dir)
    file_list.sort()
    print(file_list)

    for file in file_list:
        if file.startswith('.') or 'png' not in file or 'mask' in file:
            continue

        img = cv2.imread(ip_dir + '/' + file, 0)
        resized_img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        resized_img = np.expand_dims(resized_img, axis=0)

        x = torch.from_numpy(np.expand_dims(resized_img, axis=0) / 255)

        t = np.linspace(0, 1, num=num_pts, endpoint=True)
        t = np.expand_dims(t, axis=0)
        t = torch.from_numpy(np.expand_dims(t, axis=0))

        with torch.no_grad():
            x = x.to(device=mydevice, dtype=torch.float)
            t = t.to(device=mydevice, dtype=torch.float)

            pred_img, pred_xy = net(x, t)

            pred_xy = pred_xy.detach().cpu().numpy()

            op_png_name = '{}/{}'.format(op_dir, file)
            print(op_png_name)

            draw_pts(img, pred_xy[0], op_png_name)


if __name__ == '__main__':
    ip_dir = '/home/chentyt/Documents/4tb/Tiana/Centreline_annotation/LAO_Straight/round2/bef_predict_raw'
    op_dir = '/home/chentyt/Documents/4tb/Tiana/Centreline_annotation/LAO_Straight/round2/pred_train'
    checker()
