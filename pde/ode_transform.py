import torch
from torchdiffeq import odeint
import gc

def diric(u,N=256):
    output = torch.sin(torch.pi * u) / (N * torch.tan(torch.pi*u/N))
    output = output.nan_to_num(nan=1.,posinf = 1., neginf = 1.)
    return output


def xtu_resample(xtu,nt,nx,return_support = False):
    b = xtu.shape[0]

    x = xtu[...,0]
    t = xtu[...,1]
    u = xtu[...,2]

    ix = (x%1) * nx
    it = t * nt+1
    
    it_low = torch.floor(it).long();
    it_high = it_low + 1;

    torch.clamp(it_low,0,nt+1,out=it_low)
    torch.clamp(it_high,0,nt+1,out=it_high)

    w_low = it_high - it
    w_high = it - it_low

    grid_x = torch.arange(nx,device=xtu.device,dtype=torch.float32)

    diric_x = diric(ix[:,:,None] - grid_x[None,None,:],N = nx)

    out = torch.zeros(size=(b,nt+2,nx),device=xtu.device,dtype=torch.float32)
    out.scatter_add_(1, it_low[:,:,None].repeat(1,1,nx),w_low[:,:,None] * diric_x * u[:,:,None])
    out.scatter_add_(1, it_high[:,:,None].repeat(1,1,nx),w_high[:,:,None] * diric_x * u[:,:,None])

    out = out.view(b,nt+2,nx)
    out = out[:,1:-1,:].contiguous()


    if return_support:
        with torch.no_grad():
            supp = torch.zeros(size=(b,nt+2,nx),device=xtu.device,dtype=torch.float32)
            supp.scatter_add_(1, it_low[:,:,None].repeat(1,1,nx),w_low[:,:,None] * diric_x)
            supp.scatter_add_(1, it_high[:,:,None].repeat(1,1,nx),w_high[:,:,None] * diric_x)

            supp = supp.view(b,nt+2,nx)
            supp = supp[:,1:-1,:].contiguous()
        
        return out,supp
    return out

def xtu_resample_bilinear(xtu,nt,nx,return_support = False):
    b,n_grid,_ = xtu.shape
        
    x = xtu[...,0]
    t = xtu[...,1]
    u = xtu[...,2]

    ix = x * nx
    it = t * nt + 1 # set t index from -1 to nt+1 then slice it into 0 to nt

    with torch.no_grad():
        it_nw = torch.floor(it).long();
        ix_nw = torch.floor(ix).long();
        it_ne = it_nw + 1;
        ix_ne = ix_nw;
        it_sw = it_nw;
        ix_sw = ix_nw + 1;
        it_se = it_nw + 1;
        ix_se = ix_nw + 1;

    x_periodic = True
            
    nw = (it_se - it)    * (ix_se - ix)
    ne = (it    - it_sw) * (ix_sw - ix)
    sw = (it_ne - it)    * (ix    - ix_ne)
    se = (it    - it_nw) * (ix    - ix_nw)

    with torch.no_grad():
        torch.clamp(it_nw, 0, nt+1, out=it_nw)
        torch.clamp(it_ne, 0, nt+1, out=it_ne)
        torch.clamp(it_sw, 0, nt+1, out=it_sw)
        torch.clamp(it_se, 0, nt+1, out=it_se)
        
        if x_periodic:
            ix_nw = ix_nw % nx
            ix_ne = ix_ne % nx
            ix_sw = ix_sw % nx
            ix_se = ix_se % nx

    out = torch.zeros(size=(b,(nt+2)*nx),device = u.device,dtype = torch.float32)

    out.scatter_add_(1, (it_nw * nx + ix_nw).view(b,n_grid).long(), (u * nw).view(b,n_grid)).contiguous()
    out.scatter_add_(1, (it_ne * nx + ix_ne).view(b,n_grid).long(), (u * ne).view(b,n_grid)).contiguous()
    out.scatter_add_(1, (it_sw * nx + ix_sw).view(b,n_grid).long(), (u * sw).view(b,n_grid)).contiguous()
    out.scatter_add_(1, (it_se * nx + ix_se).view(b,n_grid).long(), (u * se).view(b,n_grid)).contiguous()

    out = out.view(b,nt+2,nx)
    out = out[:,1:-1,:].contiguous()
    if return_support:
        with torch.no_grad():
            supp = torch.zeros(size=(b,(nt+2)*nx),device = u.device,dtype = torch.float32)

            supp.scatter_add_(1, (it_nw * nx + ix_nw).view(b,n_grid).long(), nw.view(b,n_grid)).contiguous()
            supp.scatter_add_(1, (it_ne * nx + ix_ne).view(b,n_grid).long(), ne.view(b,n_grid)).contiguous()
            supp.scatter_add_(1, (it_sw * nx + ix_sw).view(b,n_grid).long(), sw.view(b,n_grid)).contiguous()
            supp.scatter_add_(1, (it_se * nx + ix_se).view(b,n_grid).long(), se.view(b,n_grid)).contiguous()

            supp = supp.view(b,nt+2,nx)
            supp = supp[:,1:-1,:].contiguous()
        return out,supp
    return out



def no_transform(loader,nx,nt):
    data_list = []
    for batch_idx,(u,) in enumerate(loader):
        data_list.append(u)
        
    data_transformed = torch.cat(data_list,dim=0)
    t_eff = torch.tensor([[0,nt]],device='cpu').repeat(data_transformed.shape[0],1)
    reject = torch.tensor([False],device='cpu').repeat(data_transformed.shape[0])
    return data_transformed, t_eff, reject

def transform(loader,
              delta_model,
              nx,
              nt,
              sigma,
              u_scaler,
              device='cpu',
              step_size = 0.1,
              threshold = 0.8,
              n_delta = None,
              resample = 'diric'):
    
    
    data_list = []
    t_eff_list = []
    reject_list = []
    n_delta = delta_model.n_delta if n_delta == None else n_delta
    for batch_idx,(u,) in enumerate(loader):

        u = u.to(device=device,dtype=torch.float32)
        n_data = u.shape[0]
        x = torch.arange(nx).to(device,torch.float32)/nx
        t = torch.arange(nt).to(device,torch.float32)/nt
        u = u_scaler.scale(u)
        xtu = torch.stack([x[None,None,:].repeat(n_data,nt,1),
                    t[None,:,None].repeat(n_data,1,nx),
                    u
                    ],dim=3).flatten(0,2)
        x = xtu[...,0]
        t = xtu[...,1]
        u = xtu[...,2]
        # transform the pde solution

        alpha = torch.rand(size=(n_data,n_delta),device=device) * 2 - 1
        alpha = alpha * sigma[None,:]

        def ode_func(t,x):
            delta = delta_model(x)
            delta = delta.view(n_data,nt*nx,3,n_delta)
            delta = torch.einsum('ij,iabj->iab',alpha,delta)
            delta = delta.flatten(0,1)
            return delta

        interval = torch.tensor([0,1],dtype=torch.float32,device=device)
        with torch.no_grad():
            xtu_transformed = odeint(ode_func,
                                    xtu,
                                    interval,
                                    method = 'rk4',options = {'step_size' : step_size})[1]

        xtu_transformed = xtu_transformed.view(n_data,nt*nx,3)
        if resample == 'diric':
            u_transformed,supp = xtu_resample(xtu_transformed,nt,nx,return_support = True)
        elif resample == 'bilinear':
            u_transformed,supp = xtu_resample_bilinear(xtu_transformed,nt,nx,return_support = True)
        else:
            assert False
        
        u_transformed = u_scaler.inv_scale(u_transformed)
        supp_b = supp > threshold


        t_eff = []
        reject = []
        for i in range(n_data):
            t_eff_region = torch.nonzero(supp_b[i].all(dim=1))
            rj = t_eff_region.shape[0]<60
            if rj:
                t_min = 0
                t_max = 0
            else:
                t_min = t_eff_region.min().item()
                t_max = t_eff_region.max().item()+1
            t_eff.append([t_min,t_max])
            reject.append(rj)

        data_list.append(u_transformed.cpu())
        t_eff_list.append(torch.tensor(t_eff))
        reject_list.append(torch.tensor(reject))
        gc.collect()
        torch.cuda.empty_cache()
        
    data_transformed = torch.cat(data_list,dim=0)
    t_eff = torch.cat(t_eff_list,dim=0)
    reject = torch.cat(reject_list,dim=0)
    return data_transformed, t_eff, reject



