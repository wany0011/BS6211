import numpy as np
from contextlib import redirect_stdout
from datetime import datetime
import os

import torch.cuda

import Config
from Dataset.DataSet_old import DataSet
from SystemLogs import SystemLogs
from Networks import FinalModel
# from Networks import *
from SupervisedTrainer import SupervisedTrainer
from Evaluator import Evaluator
from Device import mydevice
from Model_IO import training_model

# load parameters
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
neuron_list_b4_t = Config.neuron_list_b4_t
neuron_list_after_t = Config.neuron_list_after_t
arch = Config.arch

batch_size = Config.batch_size
num_pts = Config.num_pts
img_loss_w = Config.img_loss_w

encoder = Config.encoder

load_weight = Config.load_weight


def manual_seed():
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # print(torch.backends.cudnn.deterministic)
    # print(torch.backends.cudnn.benchmark)
    np.random.seed(seed)


def main():
    # torch.cuda.set_device('cuda:1')
    torch.cuda.empty_cache()
    SystemLogs(mydevice)  # print the hostname, pid, device etc

    manual_seed()  # seed so every run is exactly the same

    # train_data_name = 'train_aug_50+2'
    train_data_name = 'train'
    print('======== loading frames ========')

    # Aug data done outside of model.

    train_set = DataSet(ip_dir, train_data_name, num_pts, n_downsampling)

    # train_set = DataSet(ip_dir, train_data_name, num_pts, random_pts=random_pts,
    #                     aug=None, rot_flip=True, jit=True)

    valid_set = DataSet(ip_dir, 'valid', num_pts, n_downsampling)
    test_set = DataSet(ip_dir, 'test', num_pts, n_downsampling)

    net = FinalModel(n_channel_2d=n_channel_2d,
                     n_downsampling=n_downsampling,
                     out_kernel=out_kernel,
                     in_kernel=in_kernel,
                     neuron_list_b4_t=neuron_list_b4_t,
                     neuron_list_after_t=neuron_list_after_t,
                     n_pts=num_pts)

    # for name, param in net.named_parameters():
    #     print('name: ', name)
    #     print(type(param))
    #     print('param.shape: ', param.shape)
    #     print('param.requires_grad: ', param.requires_grad)
    #     print('=====')

    trainer = SupervisedTrainer(net=net, lr=lr, img_loss_w=img_loss_w)
    trainer.net.to(mydevice)  # net.to(mydevice) not working

    eva = Evaluator(img_loss_w)

    weight_path = '{}{}_{}_weight'.format(op_dir, model_id, arch)
    log_path = '{}{}_{}_log'.format(op_dir, model_id, arch)

    start_epoch = 0

    if load_weight:
        if os.path.exists(weight_path):
            checkpoint = torch.load(weight_path)
            trainer.net.load_state_dict(checkpoint['net_state_dict'])
            trainer.net.to(mydevice)    # need to load net.parameters() to gpu, before next line
            trainer.opt.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1

    # summary(net, [(1, 512, 512), (1, 1)])
    # print(trainer.net)
    # print(trainer.opt)
    # sys.exit()

    with open(log_path, 'a') as log:
        log.write(str(datetime.now()) + '\n')
        log.write('seed {}, batch_size {}, patience {}\n'.format(seed, batch_size, patience))
        log.write('input folder {}, train data {}\n'.format(ip_dir, train_data_name))
        log.write('batch_size {}, num_pts {}\n'.format(batch_size, num_pts))
        log.write('img loss w {}, neuron_list {} {} \n'.format(img_loss_w, neuron_list_b4_t, neuron_list_after_t))
        with redirect_stdout(log):
            print(trainer.net)
            print(trainer.opt)

    training_model(start_epoch, epoch_no, batch_size, patience, trainer,
                   eva, train_set, valid_set, test_set, weight_path, log_path)


if __name__ == '__main__':
    main()
