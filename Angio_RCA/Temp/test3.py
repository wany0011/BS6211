import numpy as np
import cv2

import torch.cuda

import Config
from Dataset.DataSet_old import DataSet
from SystemLogs import SystemLogs
# from Networks import *
from Device import mydevice

# load parameters
ip_dir = Config.ip_dir
op_dir = Config.op_dir
# seed = Config.seed
seed = 123456789
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


def manual_seed():
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # print(torch.backends.cudnn.deterministic)
    # print(torch.backends.cudnn.benchmark)
    np.random.seed(seed)


def draw_pts(img_, xy, points):

    for pt in range(points):

        row = xy[pt, 0]
        col = xy[pt, 1]

        img_[row-2:row+3, col, :] = [255, 0, 0]
        img_[row, col-2:col+3, :] = [255, 0, 0]

    return img_


def main():
    # torch.cuda.set_device('cuda:1')
    torch.cuda.empty_cache()
    SystemLogs(mydevice)  # print the hostname, pid, device etc

    manual_seed()  # seed so every run is exactly the same

    train_data_name = 'train_0-5'
    print('======== loading frames ========')
    # train_set = DataSet(ip_dir, train_data_name, num_pts, random_pts=False,
    #                     aug=[rotation_range, shear_range, crop_range], jit=False, get_id=True)

    train_set = DataSet(ip_dir, train_data_name, num_pts, random_pts=False,
                        aug=None, rot_flip=True, jit=False, get_id=True)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False)

    for i in range(epoch_no):
        for batch, (x, t, label_xy, id_) in enumerate(train_loader):
            print(batch, id_)
            x = x.detach().cpu().numpy()
            label_xy = label_xy.detach().cpu().numpy()[0, ...]
            print(id_, np.min(x), np.max(x), label_xy.shape)

            # print(label_xy)

            img = cv2.cvtColor(np.uint8(x[0, 0, ...] * 255), cv2.COLOR_GRAY2BGR)
            img1 = draw_pts(np.copy(img), (128 * label_xy).astype(int), points=16)
            # print(img.shape, img1.shape)
            img_all = np.concatenate((img, img1), axis=1)
            cv2.imwrite('{}{}_{}_{}.png'.format(op_png_dir, id_[0, 0], id_[0, 1], i), img_all)


if __name__ == '__main__':
    epoch_no = 1
    op_png_dir = '/home/liuwei/Angio/Image/LAO_Main_Curved/Test_Aug/'
    main()
