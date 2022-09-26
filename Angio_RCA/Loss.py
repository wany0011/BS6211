import torch.nn as nn
import torch

class Loss:

    def __init__(self, img_loss_w):
        self.img_w = img_loss_w
        self.xy_criterion = nn.MSELoss()
        # self.img_criterion = nn.BCELoss()
        self.img_criterion = nn.BCELoss(reduction='none')

    def eval_img(self, pred_img, label_img):
        # print('reached here', pred_img, label_img)
        # return self.img_criterion(pred_img, label_img)
        img_err = torch.mean(self.img_criterion(pred_img, label_img) * (label_img * 2 + 1))
        return img_err

    @staticmethod
    def eval_xy(pred_xy, label_xy, t):
        weight = abs(t - 0.5) + 0.5
        # print(pred_xy.shape, label_xy.shape, weight.shape)
        xy_error = (weight * (pred_xy - label_xy) ** 2)
        xy_error = torch.mean(xy_error)
        return xy_error

    def eval_w_img_t(self, pred_img, pred_xy, label_img, label_xy, t):
        xy_error = self.eval_xy(pred_xy, label_xy, t)
        # print(xy_error)
        img_error = self.eval_img(pred_img, label_img)
        # print(img_error)
        return self.img_w * img_error + xy_error
        # return img_error

    def eval_all(self, pred_img, pred_xy, label_img, label_xy, t):
        # print(pred_img.shape, pred_xy.shape, label_img.shape, label_xy.shape, t.shape)
        xy_error = self.eval_xy(pred_xy, label_xy, t)
        img_error = self.eval_img(pred_img, label_img)
        return self.img_w * img_error + xy_error, img_error, xy_error
