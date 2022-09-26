import torch

class contours:

    def __init__(self,alpha):
        self.eps = 1e-10
        self.alpha = alpha

    # =======================================
    # not needed since we input coords
    def make_control_pts(self,startpt,endpt,n):
        t = torch.arange(0,n+1)/n
        t = torch.stack((t,t))
        t = torch.transpose(t,0,1)
        r = startpt + t*(endpt - startpt) #+ 0.01*torch.randn_like(t)
        #r.requires_grad = True
        return r
    # =======================================
    # compute ll tensor
    def l(self,r):
        d = r[1:,:] - r[:-1,:]
        d2 = d*d
        return torch.sqrt(torch.sum(d2,1)) # sum over x,y dimension
    # =======================================
    def g(self,val):
        invv = 1./(val+self.eps) # n+1 points
        return  invv[:-1] + invv[1:] # n segments
    # =======================================
    # compute jacobian ll
    def dldr(self,r):
        Jl = torch.autograd.functional.jacobian(self.l,r)
        Jl = torch.transpose(Jl,0,2)
        Jl = torch.transpose(Jl,1,2)
        return Jl
    # =======================================
    def dgdr(self,val,ddx,ddy):
        val2 = (val+self.eps)*(val+self.eps)
        vx = torch.div(-1.*ddx,val2)
        vy = torch.div(-1.*ddy,val2)
        Jx = torch.diag(vx)
        Jy = torch.diag(vy)
        Jg = torch.stack((Jx,Jy))
        return Jg[:,:-1,:]+Jg[:,1:,:] # shape [2,nrows,ncols]
    # =======================================
    def dfdr(self,li,Jlt,val,ddx,ddy,P0):
        Jg = self.dgdr(val,ddx,ddy)
        Jgt = torch.transpose(Jg,1,2)
        gi = self.g(val)
        df = 0.5*P0*(torch.matmul(Jgt,li)+ torch.matmul(Jlt,gi))
        return df
    # =======================================
    def f(self,r,val,P0):
        li = self.l(r)
        gi = self.g(val)
        return torch.sum(li*gi)
    # =======================================
    def dsdr(self,li,Jlt,l0):
        lt = 2*(li - l0)
        return torch.matmul(Jlt,lt)
    # =======================================
    def s(self,r,l0):
        lt = self.l(r) - l0
        return torch.sum(lt*lt)
        ''' alternative s using li's std'''
        # return torch.std(self.l(r))
    # =======================================
    def dlossdr(self,r,val,ddx,ddy,P0,l0):
        li = self.l(r)
        Jl = self.dldr(r)
        Jlt = torch.transpose(Jl,1,2)
        df = self.dfdr(li,Jlt,val,ddx,ddy,P0)
        ds = self.dsdr(li,Jlt,l0)
        return df+self.alpha*ds
    # =======================================
    def loss(self,r,val,P0,l0):
        return self.f(r,val,P0)+self.alpha*self.s(r,l0)
    # =======================================
    def minimize(self, img_obj, n_segments, nstep, lr, r_init):
        
        startpt, endpt = img_obj.boundary_points()
        r = r_init

        '''Original L0 fn'''
        l0 = torch.norm(1.0*startpt-endpt)/n_segments

        ''' Modified L0 '''
        coord = img_obj.coord
        # l0 = torch.sum(torch.norm(coord[1:, :] - coord[:-1, :], dim = -1))/list(coord.shape)[0]


        loss_l = []
        for n in range(nstep):
            _,val,ddx,ddy = img_obj.interpolate(r) # get the image gradients at r
            assert (~torch.any(val<0)),'cannot take negative pixel values'
            P0 = 0.5*(val[0]+val[-1])
            grad = self.dlossdr(r,val,ddx,ddy,P0,l0)
            grad = torch.transpose(grad,0,1)
            grad[0,:] = 0.
            grad[-1,:] = 0. # fix the end points
            r = r - lr*grad
            r = torch.clamp(r,min=0.0,max=img_obj.L-1)

            if n%10==0: 
                loss_total = self.loss(r,val,P0,l0)
                # print(n,' loss ',loss_total.item())
            loss_l.append(loss_total.item())
        return r, loss_l
    # =======================================
           



