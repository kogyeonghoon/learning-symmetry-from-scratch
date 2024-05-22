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
# parser.add_argument('--eval_at_last', type=bool, default=True, action=argparse.BooleanOptionalAction,
#                     help='evaluate at last')
parser.add_argument('--early_stopping', type=int, default=0,
                    help='early_stopping')
parser.add_argument('--n_iters', type=int, default=None,
                    help='n_iters in one epoch')
parser.add_argument('--scheduler', type=str, default='step',
                    help='scheduler')
parser.add_argument('--patience', type=int, default=30,
                    help='patience')
parser.add_argument('--split', type=int, default=4,
                    help='lr split')
parser.add_argument('--suffix', type=str, default='default',
                    help='dataset suffix')
parser.add_argument('--u_scaling', type=bool, default=False, action=argparse.BooleanOptionalAction,
                    help='u scaling lps')
parser.add_argument('--resample', type=str, default='diric',
                    help='resample method')
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
# check_directory()

experiment_name = f'exp_fno_{args.pde}'
exp_path, outfile = init_path(experiment_name,args.exp_name,args)
print_out = Writer(outfile)
print_out(f"Experiment start at: {datetime.now()}")

# Initialize equations and data augmentation
if args.pde == 'KdV':
    train_data_path = 'data/KdV_train_1024_default.h5'
    valid_data_path = 'data/KdV_valid_1024_default.h5'
    test_data_path = 'data/KdV_test_4096_default.h5'
    pde_path = 'data/KdV_default.pkl'
elif args.pde == 'KS':
    train_data_path = 'data/KS_train_1024_default.h5'
    valid_data_path = 'data/KS_valid_1024_default.h5'
    test_data_path = 'data/KS_test_4096_default.h5'
    pde_path = 'data/KS_default.pkl'
elif args.pde == 'Burgers':
    # train_data_path = 'data/Burgers_train_1024_default.h5'
    # # train_data_path = '/mnt/home/gko/LPSDA/data/Heat_train_4096_fixed.h5'
    # valid_data_path = 'data/Burgers_valid_1024_default.h5'
    # test_data_path = 'data/Burgers_test_4096_default.h5'
    # pde_path = 'data/Burgers_default.pkl'
    train_data_path = f'data/Burgers_train_1024_{args.suffix}.h5'
    # train_data_path = '/mnt/home/gko/LPSDA/data/Heat_train_4096_fixed.h5'
    valid_data_path = f'data/Burgers_valid_1024_{args.suffix}.h5'
    test_data_path = f'data/Burgers_test_4096_{args.suffix}.h5'
    pde_path = f'data/Burgers_{args.suffix}.pkl'
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
                            batch_size=1024,
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
                            batch_size=1024,
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
                time_future=args.time_future).to(device)


n_delta = args.n_delta
scale_dict = {
    'KdV':0.3463,
    'KS':0.2130,
    'Burgers':20.19,
    'KdV_exponential':0.4345,
    'KdV_cylindrical':3.6310,
}

u_scale = scale_dict[args.pde]
u_scaler = ConstantScaler(u_scale)
vfop = VectorFieldOperation()
sigma = args.sigma

delta_model = DeltaModel(vfop,n_delta).to(device)

delta_experiment_name = f'exp_{args.pde}'

c1 = (dt * nt) / (dx * nx)
c2 = u_scaler.c

def lps_delta_func(xtu):
    x = xtu[:,0] # needs actual value for only the last delta
    t = xtu[:,1]
    u = xtu[:,2]

    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)
    xs = x
    ts = t
    us = u
    delta = torch.stack([
        torch.stack([ones,zeros,zeros],dim=1), # x-translation
        torch.stack([zeros,ones,zeros],dim=1), # t-translation
        torch.stack([c1*t,zeros,c2 * ones],dim=1), # galilean boost
    ],dim=2)
    delta = vfop.normalize_delta(delta)

    for i in range(delta.shape[-1]):
        for j in range(i):
            delta[...,i] = delta[...,i] - vfop.inner_product(delta[...,j],delta[...,i]) * delta[...,j]
    return delta

def lps_delta_func_u_scaling(xtu):
    x = xtu[:,0] # needs actual value for only the last delta
    t = xtu[:,1]
    u = xtu[:,2]

    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)
    xs = x
    ts = t
    us = u
    delta = torch.stack([
        torch.stack([ones,zeros,zeros],dim=1), # x-translation
        torch.stack([zeros,ones,zeros],dim=1), # t-translation
        torch.stack([zeros,zeros,u],dim=1), # u-scaling
        torch.stack([c1*t,zeros,c2 * ones],dim=1), # galilean boost
    ],dim=2)
    delta = vfop.normalize_delta(delta)

    for i in range(delta.shape[-1]):
        for j in range(i):
            delta[...,i] = delta[...,i] - vfop.inner_product(delta[...,j],delta[...,i]) * delta[...,j]
    return delta


if args.delta_exp == 'lps':
    if args.u_scaling:
        delta_model = lps_delta_func_u_scaling
    else:
        delta_model = lps_delta_func
elif args.delta_exp != 'none':
    delta_exp_path = os.path.join(delta_experiment_name,args.delta_exp)
    delta_model.load_state_dict(torch.load(os.path.join(delta_exp_path,'deltamodel.pt')))


metric_names = ['train_loss','val_loss',
                'test_loss','test_loss_std',
                'ntest_loss','ntest_loss_std',]
metric_tracker = MetricTracker(metric_names)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print_out(f'Number of parameters: {params}')

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=args.lr)

if args.scheduler == 'step':
    milestones = np.arange(args.split) * int(args.num_epochs/args.split)
        
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_decay)
elif args.scheduler == 'plateau':
    min_lr = 1e-6
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     factor = 0.4,
                                                     patience = args.patience,
                                                     min_lr = min_lr,)


# Training loop
min_val_loss = 10e30
min_epoch = 0
test_loss, ntest_loss = 10e30, 10e30
test_loss_std, ntest_loss_std = 0., 0.
batch_size = args.batch_size
criterion = torch.nn.MSELoss(reduction="none")
terminate = False
current_lr = args.lr

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
                                                    n_delta = n_delta,
                                                    resample = args.resample,)

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
    if args.test_mode:
        n_iters = 2
    elif args.n_iters != None:
        n_iters = args.n_iters
    else:
        n_iters = data_creator.t_res * 2

    for iteration in range(n_iters):
        
        losses = []
        
        for u,t_eff in augmented_loader:
            optimizer.zero_grad()
            
            start_time = []
            
            for j in range(u.shape[0]):
                t_min,t_max = t_eff[j,0],t_eff[j,1]
                start_range = t_min
                end_range = t_max - data_creator.time_history * 2
                # start_time.append(random.choice(range(start_range,end_range,data_creator.time_history)))
                start_time.append(random.choice(range(start_range,end_range-data_creator.time_history)))
            # print(start_time)
                
            x, y = data_creator.create_data(u, start_time)
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

        if(iteration % args.print_interval == 0):
            print_out(f'Training Loss (progress: {iteration / (data_creator.t_res * 2):.2f}): {torch.mean(losses)}')
    
    print_out("Evaluation on validation dataset:")
    val_loss, _, _, _ = test(args, 
                             pde, 
                             model,
                             valid_loader, 
                             data_creator,
                             criterion, 
                             device=device,
                             print_out=print_out)
    # if args.eval_at_last:
    #     eval_epoch = args.num_epochs*0.75
    # else:
    #     eval_epoch = 0
    if args.scheduler == 'plateau':
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
            current_lr = new_lr
            min_epoch = epoch
    if (val_loss < min_val_loss):
        # print_out("Evaluation on test dataset:")
        # test_loss, test_loss_std, ntest_loss, ntest_loss_std = test(args,
        #                                                             pde,
        #                                                             model,
        #                                                             test_loader,
        #                                                             data_creator,
        #                                                             criterion,
        #                                                             device=device,
        #                                                             print_out=print_out)
        
        # test_loss, test_loss_std, ntest_loss, ntest_loss_std = test_loss.item(), test_loss_std.item(), ntest_loss.item(), ntest_loss_std.item() 
        # Save model
        torch.save(model.state_dict(), os.path.join(exp_path,'model.pt'))
        print_out(f"Saved model")
        min_val_loss = val_loss
        min_epoch = epoch
    else:
        if (args.early_stopping != 0) and (args.scheduler == 'plateau'):
            if (epoch - min_epoch > args.early_stopping) and (current_lr == min_lr):
                terminate = True
                print_out("Early stopping")
        elif (args.early_stopping != 0):
            if (epoch - min_epoch > args.early_stopping):
                terminate = True
                print_out("Early stopping")
    
    if (epoch == args.num_epochs-1) or terminate:
        model.load_state_dict(torch.load(os.path.join(exp_path,'model.pt')))
        print_out("Evaluation on test dataset:")
        test_loss, test_loss_std, ntest_loss, ntest_loss_std = test(args,
                                                                    pde,
                                                                    model,
                                                                    test_loader,
                                                                    data_creator,
                                                                    criterion,
                                                                    device=device,
                                                                    print_out=print_out)
        
        test_loss, test_loss_std, ntest_loss, ntest_loss_std = test_loss.item(), test_loss_std.item(), ntest_loss.item(), ntest_loss_std.item() 
    else:
        test_loss, test_loss_std, ntest_loss, ntest_loss_std = 1e10,1e10,1e10,1e10
    
    metric_tracker.update({
        'train_loss':torch.mean(losses).item(),
        'val_loss':val_loss.item(),
        'test_loss':test_loss,
        'test_loss_std':test_loss_std,
        'ntest_loss':ntest_loss,
        'ntest_loss_std':ntest_loss_std,
    })
    metric_tracker.aggregate()
    metric_tracker.to_pandas().to_csv(os.path.join(exp_path,'metrics.csv'))
    
    if args.scheduler == 'plateau':
        scheduler.step(val_loss.item())
    else:
        scheduler.step()
    print_out(f"current time: {datetime.now()}")
    if terminate:
        break

print_out(f"Experiment end at: {datetime.now()}")

# print(f'Test loss mean {test_loss:.4f}, test loss std: {test_loss_std:.4f}')
# print(f'Normalized test loss mean {ntest_loss:.4f}, normalized test loss std: {ntest_loss_std:.4f}')
