import torch
import itertools


class NumericalDiff_WENO():
    def __init__(self,name,prolong,target_derivative,st,sx,sst,ssx, weight_func = None):
        # given prolongations & stensils, compute numerical differentiation using least square method
        # name: name of the derivative (int or tuple)
            # if tuple: means we compute multiple derivatives at once
        # prolong: the orders of derivative to take into account when doing least square
            # e.g. if we're computing fourth derivative, we should include all smaller partials
            # should be written in (:,2) sized torch tensor
            # (:,0) entries: x-derivatives, (:,1) entries: t-derivatives
        # target_derivative: in the prolong tensor, the index of derivative we actually want to compute
            # e.g. if we're computing fourth derivative, even if we included all smaller partials, we don't want them as an output
            # should be an int or tuple -- if tuple, the length must be same as that of name
        # st, sx: width of stencil points (width of neighboring points to compute numerical differentiation on it)
            # in this implementation, we only use rectangular stencils
        # sst,ssx: substencil (must be compatible with prolong)
            
        assert (type(name) == str) or (type(name) == tuple)
        self.name = (name,) if (type(name) == str) else name
        self.prolong = prolong
        self.coeff = torch.exp(torch.lgamma(prolong[:,0]+1) + torch.lgamma(prolong[:,1]+1))

        assert (type(target_derivative) == int) or (type(target_derivative) == tuple)
        self.target_derivative = (target_derivative,) if (type(target_derivative) == int) else target_derivative
        assert len(self.name) == len(self.target_derivative)
        
        self.sx = sx
        self.st = st
        self.ssx = ssx
        self.sst = sst
        
        self.n_nbhd = (2 * sx + 1) * (2 * st + 1)
        
        self.substencils = [(i,i+ssx,j,j+sst) for i,j in itertools.product(range(2*sx +2-ssx),range(2*st +2-sst))]
        
        substencil_center = [((i0+i1)/2,(j0+j1)/2) for i0,i1,j0,j1 in self.substencils]
        if weight_func is None:
            weight_func = lambda x:1
            
        self.weight = [weight_func(x0,y0) for x0,y0 in substencil_center]
        self.weight = torch.tensor(self.weight).to(prolong.device)
        
        self.n_stencil = ssx * sst
        
    def compute_ndiff(self,xtu_sample,xtu_nbhd,nt,nx,dt,dx,u_scaler):
        delta_x = (xtu_nbhd[:,:,0] - xtu_sample[:,None,0])
        period_ind = (delta_x.abs()>0.5).any(dim=1) # for x periodic boundary
        delta_x[period_ind] = (delta_x[period_ind] + 0.5) % 1 - 0.5
        delta_x = delta_x * nx * dx
        delta_t = (xtu_nbhd[:,:,1] - xtu_sample[:,None,1]) * nt * dt

        u = xtu_nbhd[:,:,2]
        u = u_scaler.inv_scale(u)

        delta_x = delta_x.view(-1,self.sx*2+1,self.st*2+1)
        delta_t = delta_t.view(-1,self.sx*2+1,self.st*2+1)
        u = u.view(-1,self.sx*2+1,self.st*2+1)

        ss_result_dict ={n:[] for n in self.name}
        IS_list = []
        for substencil in self.substencils:
            substencil_slice = (slice(None),slice(substencil[0],substencil[1]),slice(substencil[2],substencil[3]))

            substencil_x = delta_x[substencil_slice].flatten(1,2)
            substencil_t = delta_t[substencil_slice].flatten(1,2)
            substencil_u = u[substencil_slice].flatten(1,2)
            V = substencil_x[...,None] ** self.prolong[:,0].view(1,1,-1) * substencil_t[...,None] ** self.prolong[:,1].view(1,1,-1)
            
            try:
                p = torch.linalg.solve(V,substencil_u)
                p = torch.clamp(p.nan_to_num(1),min=-1e9,max=1e9)
            except:
                V += 1e-6 * torch.eye(V.shape[-1],dtype=torch.float32,device = V.device)[None,:,:]
                p = torch.linalg.solve(V,substencil_u)
                p = torch.clamp(p.nan_to_num(1),min=-1e9,max=1e9)
                
            d = p * self.coeff[None,:]
            IS = ((d * (dt * dx) ** (self.prolong.sum(dim=1)-1)[None,:])[:,1:]**4).sum(dim=1).detach()
            
            for n,i in zip(self.name,self.target_derivative):
                ss_result_dict[n].append(d[:,i])
            IS_list.append(IS)
            del  substencil_x,substencil_t,substencil_u,V,p,d
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        IS = torch.stack(IS_list,dim=-1)
        w = 1/(IS+1e-6)**2 * self.weight[None,:]
        w = w / torch.clamp(w.sum(dim=1,keepdim=True),min=1e-6)

        result_dict=dict()
        for n,res in ss_result_dict.items():
            res = torch.stack(res,dim=-1)
            res = (res * w).sum(dim=1)
            result_dict[n] = res
            
        return result_dict


class NumericalDiffSliceCalculator():
    def __init__(self,ndiffs,device,x_periodic = True):
        # when computing multiple derivatives, we gather the neighboring points once for computational efficiency
        # ndiffs: a list of NumericalDiff object
        
        self.ndiffs = ndiffs
        self.x_periodic = x_periodic
        self.init_nbhd(device)
        
    def init_nbhd(self,device):
        sx_global = max([ndiff.sx for ndiff in self.ndiffs])
        st_global = max([ndiff.st for ndiff in self.ndiffs])
        
        x_global = torch.arange(-sx_global,sx_global+1).to(device)
        t_global = torch.arange(-st_global,st_global+1).to(device)
        n_nbhd_global = (2 * sx_global + 1) * (2 * st_global + 1)

        x_global,t_global = torch.meshgrid([x_global,t_global])
        x_global,t_global = x_global.flatten(), t_global.flatten()

        if self.x_periodic:
            base_slice = ((st_global,-st_global),(None,None))
        else:
            base_slice = ((st_global,-st_global),(sx_global,-sx_global))
        self.get_slice = lambda x: x[:,base_slice[0][0]:base_slice[0][1],base_slice[1][0]:base_slice[1][1],:]
        
        ind_nbhd_dict = dict()
        for ndiff in self.ndiffs:
            sx,st = ndiff.sx,ndiff.st
            ind_nbhd = (x_global<=sx)&(x_global>=-sx)&(t_global<=st)&(t_global>=-st)
            ind_nbhd_dict[ndiff.name] = ind_nbhd
            
        self.sx_global = sx_global
        self.st_global = st_global
        self.x_global = x_global
        self.t_global = t_global
        self.n_nbhd_global = n_nbhd_global
        self.ind_nbhd_dict = ind_nbhd_dict
        
    def __call__(self,xtu,nt,nx,dt,dx,u_scaler):
        # xtu shape: (batch_size, nt, nx, 3)
        
        xtu_sample = self.get_slice(xtu)
        xtu_sample = xtu_sample.flatten(0,2) # shape (batch_size*nt*nx,3)
        
        xtu_nbhd_global = []
        for t_,x_ in zip(self.t_global,self.x_global):
            xtu_nbhd_global.append(self.get_slice(torch.roll(xtu,shifts = (-t_,-x_),dims=(1,2))))
        xtu_nbhd_global = torch.stack(xtu_nbhd_global,dim=3)
        xtu_nbhd_global = xtu_nbhd_global.flatten(0,2) # shape (batch_size*nt*nx,n_nbhd,3)
        
        output_dict=dict()
        output_dict['x'] = xtu_sample[...,0] * nx * dx
        output_dict['t'] = xtu_sample[...,1] * nt * dt
        for ndiff in self.ndiffs:
            ind_nbhd = self.ind_nbhd_dict[ndiff.name]
            xtu_nbhd = xtu_nbhd_global[:,ind_nbhd,:]
            result_dict = ndiff.compute_ndiff(xtu_sample,xtu_nbhd,nt,nx,dt,dx,u_scaler)
            for name,result in result_dict.items():
                output_dict[name] = result
        
        return output_dict

def get_calculator(pde,device):

    def weight_func(x0,y0):
            if (x0<1)&(y0<1):
                return 100
            else:
                return 1

    ndiff0 = NumericalDiff_WENO(name = 'u',
                                prolong = torch.tensor([[0,0],
                                                        ]).to(device),
                                target_derivative = 0,
                                st = 0,
                                sx = 0,
                                sst = 1,
                                ssx = 1,
                                weight_func = weight_func
                                )
    ndiff1 = NumericalDiff_WENO(name = ('u_x','u_t'),
                                prolong = torch.tensor([[0,0],
                                                        [0,1],
                                                        [1,0],
                                                        [1,1],
                                                        ]).to(device),
                                target_derivative = (2,1),
                                st = 1,
                                sx = 1,
                                sst = 2,
                                ssx = 2,
                                weight_func = weight_func
                                )
    ndiff2 = NumericalDiff_WENO(name = 'u_xx',
                                prolong = torch.tensor([[0,0],
                                                        [0,1],
                                                        [1,0],
                                                        [1,1],
                                                        [2,0],
                                                        [2,1],
                                                        ]).to(device),
                                target_derivative = -2,
                                st = 1,
                                sx = 2,
                                sst = 2,
                                ssx = 3,
                                weight_func = weight_func
                                )
    ndiff3 = NumericalDiff_WENO(name = 'u_xxx',
                                prolong = torch.tensor([[0,0],
                                                        [0,1],
                                                        [1,0],
                                                        [1,1],
                                                        [2,0],
                                                        [2,1],
                                                        [3,0],
                                                        [3,1],
                                                        ]).to(device),
                                target_derivative = -2,
                                st = 1,
                                sx = 2,
                                sst = 2,
                                ssx = 4,
                                weight_func = weight_func
                                )
    ndiff4 = NumericalDiff_WENO(name = 'u_xxxx',
                                prolong = torch.tensor([[0,0],
                                                        [0,1],
                                                        [1,0],
                                                        [1,1],
                                                        [2,0],
                                                        [2,1],
                                                        [3,0],
                                                        [3,1],
                                                        [4,0],
                                                        [4,1],
                                                        ]).to(device),
                                target_derivative = -2,
                                st = 1,
                                sx = 2,
                                sst = 2,
                                ssx = 5,
                                weight_func = weight_func
                                )
    if pde == 'KdV':
        ndiffs = [ndiff0,ndiff1,ndiff3,]
    elif pde == 'KS':
        ndiffs = [ndiff0,ndiff1,ndiff2,ndiff4,]
    elif pde  == 'Burgers':
        ndiffs = [ndiff0,ndiff1,ndiff2,]
    elif pde == 'nKdV':
        ndiffs = [ndiff0,ndiff1,ndiff3,]
    elif pde == 'cKdV':
        ndiffs = [ndiff0,ndiff1,ndiff3,]
    else:
        assert False
        
    return NumericalDiffSliceCalculator(ndiffs,device=device,x_periodic = False)

def compute_residual(pde,partials):
    if str(pde) == 'KdV':
        u = partials['u']
        u_x = partials['u_x']
        u_t = partials['u_t']
        u_xxx = partials['u_xxx']
        pde_value = (u_t + u * u_x + u_xxx).abs()
    elif str(pde) == 'KS':
        u = partials['u']
        u_x = partials['u_x']
        u_t = partials['u_t']
        u_xx = partials['u_xx']
        u_xxxx = partials['u_xxxx']
        pde_value = (u_t + u_xx + u_xxxx + u * u_x).abs()
    elif str(pde) == 'Burgers':
        nu = pde.nu
        u = partials['u']
        u_x = partials['u_x']
        u_t = partials['u_t']
        u_xx = partials['u_xx']
        pde_value = (u_t + u * u_x - nu * u_xx).abs()
    elif str(pde) == 'nKdV':
        t = partials['t'] + (pde.tmax-pde.tmin) / (pde.nt-1) * (pde.nt - pde.nt_effective)
        u = partials['u']
        u_x = partials['u_x']
        u_t = partials['u_t']
        u_xxx = partials['u_xxx']
        pde_value = (u_t  / torch.exp(t / 50.) + u * u_x + u_xxx).abs()
    elif str(pde) == 'cKdV':
        t = partials['t'] + (pde.tmax-pde.tmin) / (pde.nt-1) * (pde.nt - pde.nt_effective)
        u = partials['u']
        u_x = partials['u_x']
        u_t = partials['u_t']
        u_xxx = partials['u_xxx']
        pde_value = (u_t + u * u_x + u_xxx + (u/(2*t+2))).abs()
    else:
        assert False
        
    return pde_value