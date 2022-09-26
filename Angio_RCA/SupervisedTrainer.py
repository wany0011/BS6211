import torch
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
from Device import mydevice
from Loss import Loss


class SupervisedTrainer:

    def __init__(self, net, lr, img_loss_w):
        self.loss = Loss(img_loss_w)
        self.net = net
        self.opt = optim.Adam(self.net.parameters(), lr)
        # added scheduler to adjust lr per epoch
        # self.scheduler = scheduler.ExponentialLR(self.opt, gamma=0.3)

    # ===============================================
    def one_epoch(self, data_loader):
        self.net.train()
        for batch, (x, label_img, t, label_xy) in enumerate(data_loader):
            self.opt.zero_grad()
            # print(batch, x.shape, t.shape)
            x = x.to(device=mydevice, dtype=torch.float)
            t = t.to(device=mydevice, dtype=torch.float)
            label_img = label_img.to(device=mydevice, dtype=torch.float)
            label_xy = label_xy.to(device=mydevice, dtype=torch.float)

            # print(x.shape, t.shape, label_img.shape, label_xy.shape)
            # with torch.cuda.amp.autocast():
            pred_img, pred_xy = self.net(x, t)

            # print(label_img.shape, pred_img.shape, pred_xy.shape)
            loss = self.loss.eval_w_img_t(pred_img, pred_xy, label_img, label_xy, t)

            # print(loss)

            loss.backward()
            self.opt.step()
        # self.scheduler.step()
