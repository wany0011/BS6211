import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from cv2 import GaussianBlur


import time

class image:

    # def __init__(self,L=128):
    def __init__(self, img, coord, invert=False):
        """
        :param img: tensor of (1, 128, 128) dtype: torch.float32
        :param coord: (no_pts, 2) dtype: torch.int64
        :param invert: boolean if img is inverted. Default is False.
        """

        self.org = torch.squeeze(img) # save original image
        self.img = img
        self.L = self.org.shape[0]
        k = self.L//5
        s = self.L/20
        self.coord = coord
        self.invert = invert

        # INVERT PIXELS #
        if not self.invert:
            self.org = 1 - self.org
            self.img = 1 - self.img
            self.invert = True

        # GAUSSIAN BLUR VIA TORCHVISION #
        data_transforms = transforms.GaussianBlur(3, sigma=(s))
        self.img = data_transforms(self.img)
        self.img = torch.clamp(self.img, min=1e-10, max = 255)

        # GAUSSIAN BLUR VIA CV2 #
        # smooth = GaussianBlur(self.org.detach().numpy(), (k, k), s)
        # smooth = np.expand_dims(smooth, axis=0)
        # self.img = torch.from_numpy(smooth).type(torch.FloatTensor)
        # self.img = torch.clamp(self.img, min=1e-10, max = 255)


        self.ddr = self.compute_ddr()
        self.ddc = self.compute_ddc()



    def boundary_points(self):
        start, end = self.coord[0, :], self.coord[-1, :]
        return start, end

    def get_val(self):
        return torch.squeeze(self.img, 0)
       
    def get_ddr(self): # get diff wrt row
        return self.ddr
       
    def get_ddc(self): # get diff wrt col
        return self.ddc

    def pad_row(self,x):
        x = torch.squeeze(x)
        top = torch.unsqueeze(x[0,:],0) # [row,col] format
        bot = torch.unsqueeze(x[-1,:],0)
        x = torch.cat((top,x),0)
        x = torch.cat((x,bot),0)
        return torch.unsqueeze(x,0)

    def pad_col(self,x):
        x = torch.squeeze(x)
        lef = torch.unsqueeze(x[:,0],1) # [row,col] format
        rig = torch.unsqueeze(x[:,-1],1)
        x = torch.cat((lef,x),1)
        x = torch.cat((x,rig),1)
        return torch.unsqueeze(x,0)

    def dd(self,w):
        if w.shape[3] == 3:
            x = self.pad_col(self.img)
        else:
            x = self.pad_row(self.img)

        b = torch.zeros([1])
        stride = 1
        padding = 0
        x = torch.unsqueeze(x, 0)
        grad = F.conv2d(x, w, b, stride, padding)
        img = torch.squeeze(grad, 0)
        assert(self.img.shape == img.shape),'error in shape after conv'
        return img

    # can also be converted via sobel kernel
    def compute_ddr(self):
        w = torch.tensor([[[[-1.],[0.],[1.]]]])
        d = self.dd(w)
        return 0.5*d

    def compute_ddc(self):
        w = torch.tensor([[[[-1.,0.,1.]]]])
        d = self.dd(w)
        return 0.5*d

    def draw(self, r, smooth=False):
        """
        :param r: input coordinates
        :param smooth: if coordinates are drawn on org or smoothed img
        :return: overlay of image and coordinates
        """
        img_overlay = self.org.clone()
        if smooth:
            img_overlay = torch.squeeze(self.img.clone())
        for dx in [0]: #[-1,0,1]:
            for dy in [0]: #[-1,0,1]:
                row = r[:,0].long()+dx # [row,col] format
                col = r[:,1].long()+dy # index go thr column first and then row
                # reshape(-1) : goes as idx = self.L*row+col
                idx = self.L*row+col
                img_overlay.reshape(-1)[idx] = 0
        return img_overlay

    def coord2index(self,coord):
        coord = torch.clamp(coord,min=0,max=self.L-1)
        return self.L*coord[:,:,0] + coord[:,:,1]

    def get_nei_pts(self,r):

        rc = r.long() # r of the form r[pt,(x,y)]
        off = torch.tensor([[0,0],[1,0],[0,1],[1,1]])
        off = torch.unsqueeze(off,1)
        rc = off+rc
        idx = self.coord2index(rc)
        return rc,idx

    def get_interpolate_w(self,r):

        rc,idx = self.get_nei_pts(r)
        dr  = r -rc
        d = torch.norm(dr,p=2,dim=2)
        eps = 1e-10
        invd = 1./(d+eps)
        sumd = torch.unsqueeze(torch.sum(invd,0),0)
        w = torch.div(invd,sumd)

        # check that w sums to one
        assert (~torch.any(torch.abs(torch.sum(w,0)-1.)>1e-5)),'interpolate chksum error'
        return idx,w

    def chk_range(self,nval,val):
        minval = torch.min(nval,0).values
        maxval = torch.max(nval,0).values
        chkmin = torch.any(nval<minval)
        chkmax = torch.any(nval>maxval)
        assert (~chkmin),'interpolate < min'
        assert (~chkmax),'interpolate > max'

    def interpolate(self,r):

        idx,w = self.get_interpolate_w(r)

        nval = self.get_val().reshape(-1)[idx]
        nddr = self.get_ddr().reshape(-1)[idx]
        nddc = self.get_ddc().reshape(-1)[idx]

        val = torch.sum(w*nval,0)
        ddr = torch.sum(w*nddr,0)
        ddc = torch.sum(w*nddc,0)

        self.chk_range(nval,val)
        self.chk_range(nddr,ddr)
        self.chk_range(nddc,ddc)
        return idx,val,ddr,ddc


# if __name__=='__main__':
   # L = 8
   # torch.manual_seed(2481)
   # img_obj = image(L)
   #plt.imsave('curve.png',img_obj.get_val(),cmap=plt.cm.gray)
   #plt.imsave('curvedx.png',img_obj.get_ddx(),cmap=plt.cm.gray)
   #plt.imsave('curvedy.png',img_obj.get_ddy(),cmap=plt.cm.gray)

   # put test code here...

