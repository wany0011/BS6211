import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from contours import contours
from image import image
from tqdm import tqdm



if __name__=='__main__':
    # LOAD FILES FROM DIRECTORY #
    ip_dir = "/home/chentyt/Documents/4tb/Tiana/P100/Evaluate/model0_C32D2O5I5FL0_6/"
    op_dir = "/home/chentyt/Documents/4tb/Tiana/P100/PostProcess/model0_C32D2O5I5FL0_6_contour_10-20_v2/"

    if not os.path.exists(op_dir):
        os.mkdir(op_dir)

    file_list = os.listdir(ip_dir)
    for npz in tqdm(file_list):
        if 'npz' not in npz:
            continue

        # LOAD IMG AND COORD FROM NPZ #
        id = npz.split('.')[0]
        file = np.load(ip_dir + npz)
        img = np.expand_dims(file.f.img, 0) # check shape: (1, H, W)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        coord_org = file.f.coord_org # original coordinates for comparing loss
        coord_org = torch.from_numpy(coord_org * 128).type(torch.FloatTensor)
        coord = file.f.coord
        coord = torch.from_numpy(coord * 128).type(torch.FloatTensor) # check shape: (H/W, 2)
        img_shape = img.shape[1]

        # CONFIG PARAMETERS #
        torch.manual_seed(2384)
        nseg = coord.shape[0]
        alpha = 0.05
        nstep = 50 #to change
        lr = 5e-3
        patience = 500

        c = contours(alpha)
        img_obj = image(img, coord)
        original = image(img, coord).draw(coord)


        # cutoff active contour only on distance > 10
        init_dis = int(id.split('_')[-1])
        if (init_dis > 20) or (init_dis <=10):
            continue

        p_count = 0
        total_loss = []

        for i in tqdm(range(1000)): #to change
            if p_count > patience:
                print("loss did not change for {} consecetive rounds.".format(patience))
                break

            coord, loss_l = c.minimize(img_obj, nseg, nstep, lr, coord)

            # Add patience for early stopping
            if total_loss:
                if abs(loss_l[-1] - total_loss[-1]) < 1:
                    p_count += 1
                else:
                    p_count = 0
            total_loss.extend(loss_l)
        total_loss = np.array(total_loss)

        #TODO:
        # - a way to speedup the process
        aft_dis = torch.norm(coord_org - coord, dim=-1)
        aft_dis = int(torch.mean(aft_dis))

        with open(op_dir + 'distance.log', 'a') as log:
            log.write('id: {}, init dis: {}, final dis: {}'.format(id, init_dis, aft_dis) + '\n')



        # SAVE IMAGE
        new = img_obj.draw(coord)
        # torch.cat((original, overlay), 0)
        id_nodis = '_'.join(id.split('_')[:-1])
        filename = "{}_{}_contour".format(id_nodis, str(aft_dis))
        plt.imsave(op_dir + filename + '.png', torch.cat((original, new), 1) , cmap=plt.cm.gray)

        # SAVE NPZ
        img = np.squeeze(img.detach().numpy())
        coord = coord.detach().numpy()
        # np.savez(op_dir + filename + '.npz', img=img, coord=coord)



        # PLOTTING AND SAVING LOSS CURVE (FOR DEBUGGING)#
        # plt.plot(total_loss)
        # plt.title('Loss Curve')
        # plt.xlabel('Rounds')
        # plt.ylabel('Loss')
        # plt.grid()
        # plt.savefig(op_dir + 'loss_{}.png'.format(id), bbox_inches='tight')
        # np.savez(op_dir + 'loss_{}.npz'.format(id), total_loss, loss=total_loss)
        # plt.show()