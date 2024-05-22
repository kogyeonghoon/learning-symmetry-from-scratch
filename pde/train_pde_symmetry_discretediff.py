import argparse
import os
import copy
import sys
import time
from datetime import datetime
import torch
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from utils import *
from deltamodel import VectorFieldOperation, DeltaModel
from weno import get_calculator, compute_residual
from PDEs import PDE, KdV, KS, Burgers

from torchdiffeq import odeint_adjoint as odeint


parser = argparse.ArgumentParser(description='Train pde symmetry')

# genearl
parser.add_argument('--exp_name', type=str, default='tmp',
                    help='exp_name')

# pde
parser.add_argument('--pde', type=str, default='KdV',
                    help='Experiment for PDE solver should be trained: [KdV, KS, Burgers]')
parser.add_argument('--train_samples', type=int, default=1024,
                    help='Number of training samples')

# model parameters & optimization
parser.add_argument('--batch_size', type=int, default=4,
                    help='Number of samples in each minibatch')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--scheduler', type=str, default='none',
                    help='lr scheduler')
parser.add_argument('--pretrained', type=str, default='none',
                    help='pretrained model')

# task-spesific
parser.add_argument('--n_delta', type=int, default=4,
                    help='number of delta')
parser.add_argument('--sigma', type=float, default=0.2,
                    help='scale of (maximum) transformation')

parser.add_argument('--weight_pde', type=float, default=1.,
                    help='loss weight of pde loss')
parser.add_argument('--weight_ortho', type=float, default=1.,
                    help='loss weight of ortho loss')
parser.add_argument('--weight_lipschitz', type=float, default=1.,
                    help='loss weight of lipschitz loss')
parser.add_argument('--weight_sobolev', type=float, default=1,
                    help='sobolev weight')
parser.add_argument('--n_intervals', type=int, default=4,
                    help='number of transform intervals')
parser.add_argument('--tau_lipschitz', type=float, default=3.,
                    help='tau for lipschitz loss')
parser.add_argument('--patch_size', type=int, default=64,
                    help='patch size')
parser.add_argument('--th_pde', type=float, default=0.0001,
                    help='threshold (constant threshold)')
parser.add_argument('--dataset', type=str, default=None,
                    help='dataset')
parser.add_argument('--resample', type=str, default='bilinear',
                    help='resample method')
# misc
parser.add_argument('--test_mode', type=bool, default=False, action=argparse.BooleanOptionalAction,
                    help='test mode')
parser.add_argument('--checkpoint', type=bool, default=True, action=argparse.BooleanOptionalAction,
                    help='save checkpoints')

args = parser.parse_args()


batch_size = args.batch_size
n_delta = args.n_delta

# dataset
device = torch.device('cuda')

# Initialize equations and data augmentation
if args.pde == 'KdV':
    data_path = 'data/KdV_train_1024_default.h5'
    pde_path = 'data/KdV_default.pkl'
elif args.pde == 'KS':
    data_path = 'data/KS_train_1024_default.h5'
    pde_path = 'data/KS_default.pkl'
elif args.pde == 'Burgers':
    data_path = 'data/Burgers_train_1024_default.h5'
    pde_path = 'data/Burgers_default.pkl'
elif args.pde == 'nKdV':
    data_path = 'data/nKdV_train_1024_default.h5'
    pde_path = 'data/nKdV_default.pkl'
elif args.pde == 'cKdV':
    data_path = 'data/cKdV_train_1024_default.h5'
    pde_path = 'data/cKdV_default.pkl'
else:
    raise Exception("Wrong experiment")

if args.dataset is not None:
    data_path = args.dataset

experiment_name = f'exp_{args.pde}'
exp_path, outfile = init_path(experiment_name,args.exp_name,args,subdirs = ['checkpoints'])
print_out = Writer(outfile)
print_out(f"Experiment start at: {datetime.now()}")

with open(pde_path,'rb') as f:
    pde = pickle.load(f)
    
nt = pde.nt_effective
nx = pde.nx
dt = pde.dt
dx = pde.dx

train_dataset = HDF5Dataset(data_path,
                            mode='train',
                            nt=nt,
                            nx=nx,
                            n_data=args.train_samples,
                            pde=pde)
train_loader = DataLoader(train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=4,
                            persistent_workers=True,
                            pin_memory=True)



# define model & learners

sigma = args.sigma
epochs = args.epochs
weight_pde = args.weight_pde
weight_ortho = args.weight_ortho
weight_lipschitz = args.weight_lipschitz
weight_sobolev = args.weight_sobolev
n_intervals = args.n_intervals

tau = args.tau_lipschitz
patch_size = args.patch_size
th = args.th_pde


metric_names = ['loss',
                'loss_pde','loss_pde_each',
                'loss_ortho','loss_ortho_each',
                'loss_lipschitz','loss_lipschitz_each',
                'loss_sobolev','loss_sobolev_each']

metric_tracker = MetricTracker(metric_names)
min_loss = 10e10

scale_dict = {
    'KdV':0.3463,
    'KS':0.2130,
    'Burgers':20.19,
    'nKdV':0.4345,
    'cKdV':3.6310,
}

u_scale = scale_dict[args.pde]
u_scaler = ConstantScaler(u_scale)

vfop = VectorFieldOperation()
delta_model = DeltaModel(vfop,n_delta = n_delta).to(device)

if args.pretrained != 'none':
    delta_model.load_state_dict(torch.load(os.path.join(experiment_name,args.pretrained,'deltamodel.pt')))


optimizer = optim.Adam(delta_model.parameters(), lr=args.lr)
optimizer.zero_grad()
if args.scheduler == 'step':
    milestones = [int(epochs*0.5)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones = milestones,
                                               gamma = 0.1)

random_sign = RandomSign(n_delta,device)
get_patch = GetPatch(nx = nx,nt = nt, patch_size = patch_size)

compute_lipschitz_loss = LipschitzLoss(tau=tau)
compute_sobolev_loss = SobolevLoss(k=2,nx=nx,dx=dx)


def discrete_diff(u,d,dim, periodic = True):
    u_plus = torch.roll(u,-1,dims=dim)
    u_minus = torch.roll(u,1,dims=dim)
    du = (u_plus - u_minus) / (2 * d)
    if not periodic:
        boundary_slice = [slice(None),]*len(u.shape)
        boundary_slice[dim] = [0,-1]
        du[boundary_slice] = torch.nan
    return du

def fft_diff(u,dx,dim, order = 1):

    N = u.shape[dim]
    L = dx * N 
    k = torch.arange(0,N,dtype = torch.float32).to(u.device)
    k[(N+1)//2:] = k[(N+1)//2:]-N
    
    
    if (order %2 == 1) & (N % 2 == 0):
        k[N//2] = 0

    coeff = torch.pow(2j *torch.pi * k / L,order)

    f = torch.fft.fft(u,dim=dim)
    coeff_shape = [1 if i!=dim else -1 for i in range(len(u.shape))]
    df = f * coeff.view(coeff_shape)
    du = torch.fft.ifft(df,dim=dim)
    return du.real

def discrete_multi_diff(u,dx,order,acc=2):
    # acc: even number
    m = order
    n = acc
    p = (m-1)//2 + n//2
    A = torch.arange(-p,p+1,dtype=torch.float32) * dx
    A = torch.stack([A ** i for i in range(2*p+1)],dim=0)
    b = torch.zeros(size=(2*p+1,),dtype=torch.float32)
    b[m] = torch.exp(torch.lgamma(torch.tensor(m)+1))

    diff_kernel = torch.linalg.solve(A,b)
    diff_kernel = diff_kernel.view(1,1,-1).to(u.device)
    
    batch_size, nt, nx = u.shape
    u_padded = torch.cat([u[...,-p:],u, u[...,:p]],dim=2)
    u_padded = u_padded.view(-1,1,nx+2*p)

    du = F.conv1d(u_padded,diff_kernel)
    du = du.view(batch_size,nt,nx)
    return du

from ode_transform import xtu_resample_bilinear, xtu_resample

class ComputeMask():
    def __init__(self,t_neighbor,x_neighbor,device):
        assert (t_neighbor % 2 == 1) & (x_neighbor % 2 == 1)
        self.t_neighbor = t_neighbor
        self.x_neighbor = x_neighbor
        self.t_half = t_neighbor // 2
        self.x_half = x_neighbor // 2
        self.device = device
        
        kernel = torch.zeros(size=(t_neighbor,x_neighbor))
        kernel[self.t_half,:] = 1
        kernel[:,self.x_half] = 1
        kernel = kernel[None,None,...].to(self.device)
        self.kernel = kernel
        
    def get_mask(self,supp,threshold = 0.95):
        
        supp_padded = torch.clamp(supp, max=1)
        supp_padded = torch.cat([supp_padded[:,-self.t_half:,:],supp_padded, supp_padded[:,:self.t_half,:]],dim=1)
        supp_padded = torch.cat([supp_padded[:,:,-self.x_half:],supp_padded, supp_padded[:,:,:self.x_half]],dim=2)
        supp_padded = supp_padded[:,None,:,:]

        return F.conv2d(supp_padded,self.kernel).view(supp.shape) > self.kernel.sum()*threshold


compute_mask = ComputeMask(t_neighbor = 3, x_neighbor = 2*4+3, device = device)


for epoch in range(epochs):
    for batch_idx,(u,) in tqdm(enumerate(train_loader)):
        u = u.to(device)

        batch_size,_,_ = u.shape
        
        x = torch.arange(nx).to(device,torch.float32)/nx
        t = torch.arange(nt).to(device,torch.float32)/nt
        u = u_scaler.scale(u)
        xtu = torch.stack([x[None,None,:].repeat(batch_size,nt,1),
                    t[None,:,None].repeat(batch_size,1,nx),
                    u
                    ],dim=3)
        patch_x = torch.randint(0,nx,size=(batch_size,))
        patch_t = torch.randint(0,nt-patch_size,size=(batch_size,))

        xtu_patch = get_patch(xtu,patch_x,patch_t)

        xtu = xtu.flatten(0,2)
        xtu = xtu[...,None].repeat(1,1,n_delta)
        delta = delta_model(xtu)
        
        random_sign.set_sign()
        xtu_patch = xtu_patch.flatten(0,2)
        xtu_patch = xtu_patch[...,None].repeat(1,1,n_delta)

        alpha = np.random.rand() * sigma
        interval = torch.linspace(0.,alpha,n_intervals+1).to(device,torch.float32)
        func = lambda t,x:random_sign.apply(delta_model(x))
        xtu_transformed = odeint(func,
                                xtu_patch,
                                interval,
                                adjoint_params=delta_model.parameters(),
                                method = 'rk4',options = {'step_size' : sigma/10.})[1:]
                
        
        # EDIT starts

        xtu_transformed = xtu_transformed.view(n_intervals * batch_size,patch_size**2,3,n_delta).permute(0,3,1,2).flatten(0,1) # to shape (n_intervals * batch_size * n_delta, nt * nx, n_delta)
        if args.resample == 'bilinear':
            u_transformed,supp = xtu_resample_bilinear(xtu_transformed,nt,nx,return_support = True)
        elif args.resample == 'diric':
            u_transformed,supp = xtu_resample(xtu_transformed,nt,nx,return_support = True)
        else:
            assert False
        u_transformed = u_scaler.inv_scale(u_transformed)
        
        if args.pde == 'KS':
            
            u_x = discrete_multi_diff(u_transformed,dx,order=1,acc=2)
            u_xx = discrete_multi_diff(u_transformed,dx,order=2,acc=2)
            u_xxxx = discrete_multi_diff(u_transformed,dx,order=4,acc=2)
            u_t = discrete_diff(u_transformed,dt,dim=1,periodic = False)
            pde_value = u_t + u_xx + u_xxxx + u_transformed * u_x
        else:
            assert False
        
        # pde loss
        for th in 0.8 - np.arange(10)/10.:
            mask = compute_mask.get_mask(supp,threshold = th)
            if (mask.sum(dim=(1,2)) != 0).all():
                break
        pde_value_sum = (torch.clamp(pde_value.nan_to_num().abs(),min=1e-6).log() * mask).sum(dim=(1,2)) / torch.clamp(mask.sum(dim=(1,2)),min=1)
        # if args.logscale:
        #     pde_value_sum = torch.log(torch.clamp(pde_value_sum,min=1e-6))

        pde_value_sum = pde_value_sum.view(n_intervals, batch_size,n_delta)
        loss_pde_each = pde_value_sum.mean(dim=(0,1))
        
        
        # EDIT ends
        
        # ortho loss
        loss_ortho_each = vfop.sequential_ortho_loss(delta,final='arccos')

        # lipschitz loss
        loss_lipschitz_each = compute_lipschitz_loss(delta.view(batch_size,nt,nx,3,n_delta),
                                                     xtu.view(batch_size,nt,nx,3,n_delta).detach())

        loss_sobolev_each = compute_sobolev_loss(delta.view(batch_size,nt,nx,3,n_delta)) 
        
        loss_pde = loss_pde_each.mean() * weight_pde
        loss_ortho = loss_ortho_each.mean() * weight_ortho
        loss_lipschitz = loss_lipschitz_each.mean() * weight_lipschitz
        if epoch > int(epochs*0.9):
            loss_sobolev = loss_sobolev_each.mean() * weight_sobolev
        else:
            loss_sobolev = loss_sobolev_each.mean() * 0
        loss = loss_pde + loss_ortho + loss_lipschitz + loss_sobolev

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        
        # update metrics
        metric_dict = ({
            'loss':loss.item(),
            'loss_pde':loss_pde.item(),
            'loss_pde_each':loss_pde_each.clone().detach().cpu(),
            'loss_ortho':loss_ortho.item(),
            'loss_ortho_each':loss_ortho_each.clone().detach().cpu(),
            'loss_lipschitz':loss_lipschitz.item(),
            'loss_lipschitz_each':loss_lipschitz_each.clone().detach().cpu(),
            'loss_sobolev':loss_sobolev.item(),
            'loss_sobolev_each':loss_sobolev_each.clone().detach().cpu(),
        })
        metric_tracker.update(metric_dict)
        
        if args.test_mode:
            if batch_idx == 2:
                break
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    # gather metrics
    metric_dict = metric_tracker.aggregate()
    
    # record
    outstr = 'epoch {}, loss: {:.4f}, loss_pde {:.4f}, loss_ortho {:.4f}, loss_lipschitz {:.4f}, loss_sobolev {:.4f}\n'.format(epoch,metric_dict['loss'],metric_dict['loss_pde'],metric_dict['loss_ortho'],metric_dict['loss_lipschitz'],metric_dict['loss_sobolev'])
    outstr += 'loss_pde for each:\n' + str(metric_dict['loss_pde_each']) + '\n'
    outstr += 'loss_ortho for each:\n' + str(metric_dict['loss_ortho_each']) + '\n'
    outstr += 'loss_lipschitz for each:\n' + str(metric_dict['loss_lipschitz_each']) + '\n'
    outstr += 'loss_sobolev for each:\n' + str(metric_dict['loss_sobolev_each']) + '\n'
    outstr += '=' * 50
    
    # with open(outfile,'a') as f:
    #     f.write(outstr +'\n')
    print_out(outstr)
    metric_tracker.to_pandas().to_csv(os.path.join(exp_path,'metrics.csv'))
    
    # save model if loss is at minimum
    if metric_dict['loss'] < min_loss:
        min_loss = metric_dict['loss']
        torch.save(delta_model.state_dict(),os.path.join(exp_path,'deltamodel.pt'))
    
    # update scheduler
    if args.scheduler == 'step' or args.scheduler == 'cosine':
        scheduler.step()
    
    
    
    
    if args.checkpoint:
        if (epoch+1) % 10 == 0:
            torch.save(delta_model.state_dict(),os.path.join(exp_path,'checkpoints',f'epoch_{epoch}.pt'))

print_out(f"Experiment end at: {datetime.now()}")
