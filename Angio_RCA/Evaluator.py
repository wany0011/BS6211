import torch
import torch.optim as optim
from Device import mydevice
from Loss import Loss
# from sklearn.metrics import confusion_matrix


class Evaluator:
    def __init__(self, img_loss_w):
        self.loss = Loss(img_loss_w)

    def all_in_one(self, net, data_loader):
        net.eval()
        loss_img_xy = 0
        loss_img = 0
        loss_xy = 0
        dis = 0

        img_count = 0
        xy_count = 0

        with torch.no_grad():
            for batch, (x, label_img, t, label_xy) in enumerate(data_loader):

                x = x.to(device=mydevice, dtype=torch.float)
                t = t.to(device=mydevice, dtype=torch.float)
                label_img = label_img.to(device=mydevice, dtype=torch.float)
                label_xy = label_xy.to(device=mydevice, dtype=torch.float)

                # with torch.cuda.amp.autocast():
                pred_img, pred_xy = net(x, t)
                # print(output)
                loss1, loss2, loss3 = self.loss.eval_all(pred_img, pred_xy, label_img, label_xy, t)

                loss_img_xy += loss1 * pred_img.shape[0]
                loss_img += loss2 * pred_img.shape[0]
                loss_xy += loss3 * pred_img.shape[0]
                img_count += pred_img.shape[0]

                dis += torch.sum(torch.norm(pred_xy - label_xy, dim=2))
                xy_count += label_xy.shape[0] * label_xy.shape[1]

        mean_loss_img_xy = loss_img_xy / img_count
        mean_loss_img = loss_img / img_count
        mean_loss_xy = loss_xy / img_count
        mean_dis = 128 * dis / xy_count

        return mean_loss_img_xy, mean_loss_img, mean_loss_xy, mean_dis
