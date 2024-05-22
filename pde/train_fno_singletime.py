# codes from https://github.com/brandstetter-johannes/LPSDA.git

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


from typing import Tuple
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from utils import *
from fno_train_helper import *
from ode_transform import no_transform, transform
from deltamodel import VectorFieldOperation, DeltaModel

    
parser = argparse.ArgumentParser(description='Train an PDE solver')
# PDE
parser.add_argument('--device', type=str, default='cpu',
                    help='Used device')
parser.add_argument('--pde', type=str, default='KdV',
                    help='Experiment for PDE solver should be trained: [KdV, KS, Burgers, nKdV, cKdV]')
parser.add_argument('--train_samples', type=int, default=512,
                    help='Number of training samples')
parser.add_argument('--exp_name', type=str, default='tmp',
                    help='exp name')

# Model parameters
parser.add_argument('--n_delta', type=int, default=4,
                    help='n_delta')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Number of samples in each minibatch')
parser.add_argument('--transform_batch_size', type=int, default=32,
                    help='transform batch size')
parser.add_argument('--delta_exp', type=str, default='none',
                    help='delta exp name')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--lr_decay', type=float,
                    default=0.4, help='multistep lr decay')
parser.add_argument('--sigma', nargs='+',type=float,
                    default=[0,0,0,0])
parser.add_argument('--n_transform', type=int, default=1, 
                    help='number of transform')
parser.add_argument('--test_mode', type=bool, default=False, action=argparse.BooleanOptionalAction,
                    help='test mode')
parser.add_argument('--p_original', type=float, default=0,
                    help='prob of using original data')

parser.add_argument('--pred_t',type=int, default=90,
                    help='pred_t')
# Misc
parser.add_argument('--time_history', type=int,
                    default=20, help="Time steps to be considered as input to the solver")
parser.add_argument('--time_future', type=int,
                    default=20, help="Time steps to be considered as output of the solver")
parser.add_argument('--print_interval', type=int, default=70,
                    help='Interval between print statements')

# args = parser.parse_args('--device cuda --batch_size 16'.split(' '))
args = parser.parse_args()

if args.test_mode:
    args.n_transform = min([2,args.n_transform])


device = args.device

experiment_name = f'exp_fno_{args.pde}'
exp_path, outfile = init_path(experiment_name,args.exp_name,args)
print_out = Writer(outfile)
print_out(f"Experiment start at: {datetime.now()}")

# check_directory()
# Initialize equations and data augmentation
if args.pde == 'nKdV':
    train_data_path = 'data/nKdV_train_1024_default.h5'
    valid_data_path = 'data/nKdV_valid_1024_default.h5'
    test_data_path = 'data/nKdV_test_4096_default.h5'
    pde_path = 'data/nKdV_default.pkl'
elif args.pde == 'cKdV':
    train_data_path = 'data/cKdV_train_1024_default.h5'
    valid_data_path = 'data/cKdV_valid_1024_default.h5'
    test_data_path = 'data/cKdV_test_4096_default.h5'
    pde_path = 'data/cKdV_default.pkl'
else:
    raise Exception("Wrong experiment")

with open(pde_path,'rb') as f:
    pde = pickle.load(f)

nt = pde.nt_effective
nx = pde.nx
dt = pde.dt
dx = pde.dx

train_dataset = HDF5Dataset(train_data_path,
                            mode='train',
                            nt=nt,
                            nx=nx,
                            n_data = args.train_samples,
                            pde=pde,)
train_loader = DataLoader(train_dataset,
                            batch_size=args.transform_batch_size, # not args.batch_size!
                            shuffle=False,
                            num_workers=4,
                            persistent_workers=True,
                            pin_memory=True)

valid_dataset = HDF5Dataset(valid_data_path,
                            mode='valid',
                            nt=nt,
                            nx=nx,
                            pde=pde,)
valid_loader = DataLoader(valid_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=4,
                            persistent_workers=True,
                            pin_memory=True)

test_dataset = HDF5Dataset(test_data_path,
                            mode='test',
                            nt=nt,
                            nx=nx,
                            pde=pde,)
test_loader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=4,
                            persistent_workers=True,
                            pin_memory=True)


# # Initialize DataCreator and model
data_creator = DataCreator(time_history=args.time_history,
                            time_future=args.time_future,
                            t_resolution=nt,
                            x_resolution=nx
                            ).to(device)
model = FNO1d(pde=pde,
                time_history=args.time_history,
                time_future=1).to(device)


n_delta = args.n_delta
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
sigma = args.sigma

delta_model = DeltaModel(vfop,n_delta).to(device)

if str(pde) == 'nKdV':

    c1 = (dt * nt) / (dx * nx)
    c2 = u_scaler.c
    def lps_delta_func(xtu):
        x = xtu[:,0] # needs actual value for only the last delta
        t = xtu[:,1]
        u = xtu[:,2]
        t = t * nt * dt + 100 / 249 * 110

        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)
        

        delta = torch.stack([
            torch.stack([ones,zeros,zeros],dim=1),
            torch.stack([zeros,1/torch.exp(t / 50),zeros],dim=1),
            torch.stack([50 * (torch.exp(t/50)-1) / (dx*nx),zeros,c2 * ones],dim=1),
        ],dim=2)
        delta = vfop.normalize_delta(delta)

        
        # delta[...,-1] = vfop.normalize_delta(delta[...,-1] - vfop.inner_product(delta[...,0],delta[...,-1]) * delta[...,0])
        for i in range(delta.shape[-1]):
            for j in range(i):
                delta[...,i] = delta[...,i] - vfop.inner_product(delta[...,j],delta[...,i]) * delta[...,j]
            delta[...,i] = vfop.normalize_delta(delta[...,i])
        return delta

elif str(pde) == 'cKdV':

    c1 = (dt * nt) / (dx * nx)
    c2 = u_scaler.c
    def lps_delta_func(xtu):
        x = xtu[:,0] # needs actual value for only the last delta
        t = xtu[:,1]
        u = xtu[:,2]
        x = x * nx * dt
        t = t * nt * dt + 100 / 249 * 110
        u = u_scaler.inv_scale(u)

        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)
        

        delta = torch.stack([
            torch.stack([ones,zeros,zeros],dim=1),
            torch.stack([((t+1)**0.5) / (dx * nx),zeros ,1 / (2 * ((t+1) ** 0.5)) * c2],dim=1),
            torch.stack([zeros,ones,zeros],dim=1),
            # torch.stack([x / (dx * nx),3 * t / (dt * nt),-2 * u * c2 ],dim=1),
            # torch.stack([x * (t**0.5)/(dx*nx),2 * (t ** 1.5) / (dt * nt), 0.5 *(x/(t**0.5) - 4 * u * (t**0.5))*0.5 ],dim=1),
        ],dim=2)
        delta = vfop.normalize_delta(delta)

        
        for i in range(delta.shape[-1]):
            for j in range(i):
                delta[...,i] = delta[...,i] - vfop.inner_product(delta[...,j],delta[...,i]) * delta[...,j]
            delta[...,i] = vfop.normalize_delta(delta[...,i])
        # delta[...,-1] = vfop.normalize_delta(delta[...,-1] - vfop.inner_product(delta[...,0],delta[...,-1]) * delta[...,0])
        return delta

delta_experiment_name = f'exp_{args.pde}'

if args.delta_exp not in  ['none','lps']:
    delta_exp_path = os.path.join(delta_experiment_name,args.delta_exp)
    delta_model.load_state_dict(torch.load(os.path.join(delta_exp_path,'deltamodel.pt')))
elif args.delta_exp == 'lps':
    print('lps')
    delta_model = lps_delta_func

metric_names = ['train_loss','val_loss','val_nloss',
                'test_loss','test_nloss',]
metric_tracker = MetricTracker(metric_names)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print_out(f'Number of parameters: {params}')

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
if args.num_epochs == 20:
    milestones = [0,5,10,15]
elif args.num_epochs == 30:
    milestones = [0,8,15,23]
elif args.num_epochs == 40:
    milestones = [0,10,20,30]
else:
    assert False

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_decay)

# Training loop
min_val_loss = 10e30
test_loss, ntest_loss = 10e30, 10e30
test_loss_std, ntest_loss_std = 0., 0.
batch_size = args.batch_size
criterion = torch.nn.MSELoss(reduction="none")


n_transform = args.n_transform

data_transformed_list = []
t_eff_list = []
reject_list = []

for i in range(n_transform):
    if i==0:
        data_transformed, t_eff, reject = no_transform(train_loader,nx,nt)
    else:
        print_out(f"transforming {i}/{n_transform}")
        data_transformed, t_eff, reject = transform(train_loader,delta_model,nx,nt,
                                                    sigma = torch.tensor(sigma).to(device),
                                                    u_scaler = u_scaler,
                                                    device = device,
                                                    n_delta = n_delta,)

        gc.collect()
        torch.cuda.empty_cache()
    data_transformed_list.append(data_transformed)
    t_eff_list.append(t_eff)
    reject_list.append(reject)
    
data_transformed = torch.stack(data_transformed_list,dim=0)
t_eff = torch.stack(t_eff_list,dim=0)
reject = torch.stack(reject_list,dim=0)

augmented_dataset = AugmentedDataset(data_transformed,
                                     t_eff,
                                     reject,
                                     args.train_samples,
                                     n_transform,
                                     p_original = args.p_original)

augmented_loader = DataLoader(augmented_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=4,
                            persistent_workers=True,
                            pin_memory=True)



for epoch in range(args.num_epochs):
    
    print_out(f"Epoch {epoch}")

    model.train()
    n_iters = 100 if not args.test_mode else 2

    for iteration in range(n_iters):
        
        losses = []
        
        for u,t_eff in augmented_loader:
            optimizer.zero_grad()
            
            
            x = u[:,20:40,:]
            y = u[:,args.pred_t:args.pred_t+1,:]
            x, y = x.to(device), y.to(device)


            # Change [batch, time, space] -> [batch, space, time]
            x = x.permute(0, 2, 1)
            pred = model(x)

            loss = criterion(pred.permute(0, 2, 1), y)
            loss = loss.sum()
            loss.backward()
            losses.append(loss.detach() / x.shape[0])
            optimizer.step()
            

        losses = torch.stack(losses)
        train_loss = torch.mean(losses).item()
        if(iteration % args.print_interval == 0):
            print_out(f'Training Loss (progress: {iteration / (data_creator.t_res * 2):.2f}): {train_loss}')
        

            
    losses = []
    nlosses = []
    for (u, ) in valid_loader:
        with torch.no_grad():
            x = u[:,20:40,:]
            y = u[:,args.pred_t:args.pred_t+1,:]
            x, y = x.to(device), y.to(device)
            x = x.permute(0, 2, 1)

            pred = model(x)

            loss = criterion(pred.permute(0, 2, 1), y)
            nloss = loss.sum(dim=(1,2)) / (y**2).sum(dim=(1,2))
            loss = loss.sum()
            losses.append(loss.detach() / x.shape[0])
            nlosses.append(nloss.sum().detach() / x.shape[0])
    losses = torch.stack(losses)
    nlosses = torch.stack(nlosses)
    val_loss = torch.mean(losses).item()
    val_nloss = torch.mean(nlosses).item()
    print_out(f'Validation loss {val_loss}, normalized loss {val_nloss}')

    
    if (val_loss < min_val_loss) & (epoch>args.num_epochs*0.75):

        # Save model
        torch.save(model.state_dict(), os.path.join(exp_path,'deltamodel.pt'))
        print_out(f"Saved model")
        min_val_loss = val_loss
        
        losses = []
        nlosses = []
        for (u, ) in test_loader:
            with torch.no_grad():
                x = u[:,20:40,:]
                y = u[:,args.pred_t:args.pred_t+1,:]
                x, y = x.to(device), y.to(device)
                x = x.permute(0, 2, 1)
                pred = model(x)

                loss = criterion(pred.permute(0, 2, 1), y)
                nloss = loss.sum(dim=(1,2)) / (y**2).sum(dim=(1,2))
                loss = loss.sum()
                losses.append(loss.detach() / x.shape[0])
                nlosses.append(nloss.sum().detach() / x.shape[0])
        losses = torch.stack(losses)
        nlosses = torch.stack(nlosses)
        test_loss = torch.mean(losses).item()
        test_nloss = torch.mean(nlosses).item()
        print_out(f'Validation loss {test_loss}, normalized loss {test_nloss}')
    else:
        test_loss,test_nloss = 1e10,1e10
    
    metric_tracker.update({
        'train_loss':train_loss,
        'val_loss':val_loss,
        'val_nloss':val_nloss,
        'test_loss':test_loss,
        'test_nloss':test_nloss,
    })
    metric_tracker.aggregate()
    metric_tracker.to_pandas().to_csv(os.path.join(exp_path,'metrics.csv'))
    
    scheduler.step()
    print_out(f"current time: {datetime.now()}")

print_out(f"Experiment end at: {datetime.now()}")

# print(f'Test loss mean {test_loss:.4f}, test loss std: {test_loss_std:.4f}')
# print(f'Normalized test loss mean {ntest_loss:.4f}, normalized test loss std: {ntest_loss_std:.4f}')
