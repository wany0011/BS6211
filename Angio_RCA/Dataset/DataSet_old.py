import sys

import cv2
import numpy as np
from scipy import ndimage
from torchvision import transforms
from torch.utils.data import Dataset


class DataSet(Dataset):

    def __init__(self, file_folder, file_name, num_pts, n_downsampling, rot_flip=False,
                 jit=False, get_id=False):

        self.__num_pts = num_pts
        self.__get_pts = self.__even_pots
        self.__rot_flip = rot_flip
        self.__jit = jit
        self.__mid_size = 128 // 2 ** n_downsampling

        with np.load(file_folder + file_name + '.npz') as npz:
            self.__get_id = get_id
            if self.__get_id:
                self.__id = npz['id']

            self.__img = npz['img'].astype(np.float32)
            self.__img_size = self.__img.shape[-1]
            self.__label = npz['coord']
            self.__pt_range = self.__label.shape[-2]
            print('Total points: ', self.__pt_range, self.__label.shape)
            print('Coord range: ', np.min(self.__label), np.max(self.__label))

        print('Overall img shape: {} \t label shape: {}'.format(self.__img.shape, self.__label.shape))

    def __len__(self):
        return self.__img.shape[0]

    def len(self):
        return self.__len__()

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise ValueError('idx ' + str(idx) + ' exceed length of data: ' + str(self.__len__()))

        pts_ = self.__get_pts()

        # for input image
        img_ = self.__img[idx]
        if self.__jit:
            img_ = self._jitter(img_)
        img_ = np.expand_dims(img_, axis=0) / 255

        # for binary image
        coord_all = self.__mid_size * self.__label[idx, :, :]
        coord_all = np.rint(coord_all).astype(np.uint8)
        coord_all = np.unique(coord_all, axis=0)
        np.clip(coord_all, 0, self.__mid_size-1, coord_all)
        bw_img = np.zeros((self.__mid_size, self.__mid_size), dtype=np.uint8)
        bw_img[coord_all[:, 0], coord_all[:, 1]] = 1
        bw_img = np.expand_dims(bw_img, axis=0)

        # for xy
        coord_ = self.__label[idx, pts_, :]

        # for pts
        pts_ = pts_ / (self.__pt_range - 1)
        pts_ = np.expand_dims(pts_, axis=1)

        if self.__get_id:
            id_ = self.__id[idx, ...]
            return img_, bw_img, pts_, coord_, id_
        else:
            return img_, bw_img, pts_, coord_

    def __even_pots(self):
        pts_ = np.linspace(0, self.__pt_range-1, num=self.__num_pts, endpoint=True).astype(int)
        return pts_

    @staticmethod
    def _jitter(img):
        jitter = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
            transforms.Grayscale(num_output_channels=1),
        ])
        img = jitter(img)
        return np.array(img)
