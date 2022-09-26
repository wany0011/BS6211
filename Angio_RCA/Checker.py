import torch
import numpy as np
import sys
import cv2
import os

import Config
from matplotlib import pyplot as plt
# import resource
from Dataset.DataSet_old import DataSet
from Device import mydevice
from SystemLogs import SystemLogs
from Networks import FinalModel
# from SupervisedTrainer import SupervisedTrainer
from Loss import Loss
# from model_IO import training_model

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
neuron_list_b4_t = Config.neuron_list_b4_t
neuron_list_after_t = Config.neuron_list_after_t
arch = Config.arch

batch_size = Config.batch_size
num_pts = Config.num_pts
img_loss_w = Config.img_loss_w

encoder = Config.encoder


def draw_pts(img, label_img, pred_img, pred_xy, label_xy, op_name):
    """ Function to use pred_xy to plot over bgr_img and pred_img, then concat original, pred and label img """
    bgr_img = cv2.cvtColor(np.uint8(img * 255), cv2.COLOR_GRAY2BGR)
    label_img = cv2.cvtColor(np.uint8(label_img * 255), cv2.COLOR_GRAY2BGR)
    label_img = cv2.resize(label_img, img.shape, interpolation=cv2.INTER_AREA)
    pred_img = cv2.cvtColor(np.uint8(pred_img * 255), cv2.COLOR_GRAY2BGR)
    pred_img = cv2.resize(pred_img, img.shape, interpolation=cv2.INTER_AREA)
    cp_img = np.copy(bgr_img)
    # print(bgr_img.shape, pred_xy.shape, label_xy.shape)

    # print(np.min(label_xy * img.shape[0]), np.max(label_xy * img.shape[1]))
    # print(np.min(pred_xy * img.shape[0]), np.max(pred_xy * img.shape[1]))
    label_xy = np.clip(label_xy * img.shape[0], 3, 124).astype(int)
    pred_xy = np.clip(pred_xy * img.shape[0], 3, 124).astype(int)

    for pt in range(num_pts):

        # draw predicted coordinates (+)
        row, col = pred_xy[pt, :]

        cp_img[row-3:row+4, col, :] = [0, 0, 255]
        cp_img[row, col-3:col+4, :] = [0, 0, 255]

        # pred_img[row-3:row+4, col, :] = [0, 0, 255]
        # pred_img[row, col-3:col+4, :] = [0, 0, 255]

        # print(np.min(label_xy), np.max(label_xy))

        # draw labelled coordinates (+)
        row, col = label_xy[pt, :]

        for i in range(4):
            cp_img[row-i, col-i, :] = [255, 0, 0]
            cp_img[row-i, col+i, :] = [255, 0, 0]
            cp_img[row+i, col-i, :] = [255, 0, 0]
            cp_img[row+i, col+i, :] = [255, 0, 0]

    bgr_img = np.concatenate((bgr_img, pred_img, label_img, cp_img), axis=1)
    # print(bgr_img.shape)
    cv2.imwrite(op_name, bgr_img)
    return


def checker():
    torch.cuda.empty_cache()
    SystemLogs(mydevice)  # print the hostname, pid, device etc
    net = FinalModel(n_channel_2d=n_channel_2d,
                     n_downsampling=n_downsampling,
                     out_kernel=out_kernel,
                     in_kernel=in_kernel,
                     neuron_list_b4_t=neuron_list_b4_t,
                     neuron_list_after_t=neuron_list_after_t,
                     n_pts=num_pts)

    weight_path = '{}{}_{}_weight'.format(model_dir, model_id, arch)
    print('Loading: ', weight_path)
    if not os.path.exists(weight_path):
        sys.exit('No trained model exists.')

    op_path = '{}{}_{}'.format(op_dir, model_id, arch)
    if not os.path.exists(op_path):
        os.makedirs(op_path)

    # loading the pretrained weight onto the same model architecture
    # checkpoint = torch.load(weight_path)
    checkpoint = torch.load(weight_path,map_location=torch.device('cpu'))
    # checkpoint = torch.load("/home/chentyt/Documents/4tb/Tiana/P100/P100_RCA/archive/version")
    # print(checkpoint.items())

    net.load_state_dict(checkpoint['net_state_dict'])
    net.to(mydevice)
    net.eval()
    epoch = checkpoint['epoch']
    print('Epoch:', epoch)

    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': False} if use_cuda else {}

    loss = Loss(img_loss_w)

    for npz_name in ip_list:
        data_set = DataSet(ip_dir, npz_name, num_pts, n_downsampling, get_id=False)
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=False, **kwargs)

        dist_list = []  # list of euclidean distance metric for each npz_name (train, test or valid)
        loss_w_img = 0
        loss_no_img = 0
        count = 0

        with torch.no_grad():
            for batch, (x, label_img, t, label_xy) in enumerate(data_loader):

                # print(x.shape, t.shape, label_xy.shape, id_.shape)
                x = x.to(device=mydevice, dtype=torch.float)
                t = t.to(device=mydevice, dtype=torch.float)
                label_img = label_img.to(device=mydevice, dtype=torch.float)
                label_xy = label_xy.to(device=mydevice, dtype=torch.float)

                # pred_img output is not useful here
                pred_img, pred_xy = net(x, t)

                # metric: finding the euclidian distance between true vs predicted labels
                dis = torch.norm(pred_xy - label_xy, dim=2)

                # dis[128 * dis < 5] = 0

                total_loss, img_loss, xy_loss = loss.eval_all(pred_img, pred_xy, label_img, label_xy, t)
                loss_w_img += total_loss
                loss_no_img += img_loss
                # loss_w_img += loss.eval_w_img(pred_img, pred_xy, label_img, label_xy)
                # loss_no_img += loss.eval(pred_xy, label_xy)
                count += x.shape[0]

                # detach from cuda tensor to cpu for numpy tensor.
                x = x.detach().cpu().numpy()
                pred_img = pred_img.detach().cpu().numpy()
                pred_xy = pred_xy.detach().cpu().numpy()
                label_img = label_img.detach().cpu().numpy()
                label_xy = label_xy.detach().cpu().numpy()

                mean_dis = torch.mean(dis).detach().cpu().numpy()

                # print(dis, mean_dis, xy_loss)

                dist_list.append(int(128 * mean_dis))

                # print(id_, int(128 * mean_dis))

                if save_pic:
                    op_png_name = '{}/{}_{}_{:2f}_{}.png'.format(
                        op_path, npz_name, count, img_loss, int(mean_dis*128))

                    draw_pts(x[0, 0, :, :], label_img[0, -1, :, :],
                             pred_img[0, 0, :, :], pred_xy[0], label_xy[0], op_png_name)

                #Save image as npz format
                if save_npz:
                    # np.savez('{}/{}_{}_{}_{:.2f}_{}.npz'.format(op_path, epoch, npz_name, count, img_loss, int(mean_dis*128)), img=x[0, 0, :, :], coord=pred_xy[0], coord_org=label_xy[0])
                    np.savez('{}/{}_{}_{}_{}.npz'.format(op_path, epoch, npz_name, count, int(mean_dis * 128)), img=x[0, 0, :, :], coord=pred_xy[0], coord_org=label_xy[0])

            dist_list = np.array(dist_list)
            dist_list[dist_list > 40] = 40

            # save full array for valid/test
            # np.savez('{}/{}_{}.npz'.format(op_path, epoch, npz_name), dist_list)

            print('loss with img {}, without img {}'.format(loss_w_img / count, loss_no_img / count))
            print('Pixel distance: max {}, min {}, mean {}, std {}'.format(max(dist_list), min(dist_list),
                                                                           np.mean(dist_list), np.std(dist_list)))

            print(dist_list.shape, dist_list)
            plt.hist(dist_list, bins=8, range=(0, 40), density=True)
            # plt.show()

            # visualise distribution
            print(np.histogram(dist_list, bins=8, range=(0, 40)))


if __name__ == '__main__':
    ip_dir = '/home/chentyt/Documents/4tb/Tiana/P100/RCA_Split1/'
    # below are non-augmented data
    ip_list = ['valid', 'test']
    # ip_list = ['train', 'valid', 'test']
    # ip_list = ['train_aug_50+2']
    op_dir = '/home/chentyt/Documents/4tb/Tiana/P100/Evaluate/'
    save_pic = True
    save_npz = True
    checker()
