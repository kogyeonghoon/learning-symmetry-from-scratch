import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import math
import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from torch.autograd.functional import jvp
from tqdm import tqdm

from deltamodel import VectorFieldOperation, DeltaModel, compute_coords_weight, LipschitzLoss
from utils import grid_sample, plot_delta, plot_delta_instance, MetricTracker, plot_coords_weight

from torchdiffeq import odeint_adjoint as odeint
from resnet import resnet18


experiment_name = 'exp_ode_method'

parser = argparse.ArgumentParser()

# basic
parser.add_argument("--n_delta", type = int)
parser.add_argument("--exp_name", type = str)

# optimization
parser.add_argument("--lr", type = float, default = 1e-4)
parser.add_argument("--batch_size", type = int, default = 128)
parser.add_argument("--epochs", type = int, default = 50)

# models
parser.add_argument("--resnet_name", type = str, default = 'exp_augment/benchmark/model.pt')

# scale
parser.add_argument("--ortho_weight", type = float, default = 10.)
parser.add_argument("--ortho_loss_final", type = str, default = 'arccos')
parser.add_argument("--lipschitz_weight", type = float, default = 10.)
parser.add_argument("--sigma", type = float, default = 0.4)
parser.add_argument("--tau", type = float, default = 0.5)
parser.add_argument("--device", type = str, default = 'cpu')
parser.add_argument("--coords_weight", type = str, default = 'exp_ode_method/exp2/coords_weight.pt')
# for test
parser.add_argument("--test_mode", type = bool, default = False, action=argparse.BooleanOptionalAction)

args = parser.parse_args()
device = args.device
# torch.cuda.set_device(device)

resnet = resnet18()
resnet.load_state_dict(torch.load(args.resnet_name,map_location = device))
model = nn.Sequential(resnet.conv1,
                    resnet.conv2_x,
                    resnet.conv3_x,
                    resnet.conv4_x,
                    resnet.conv5_x,
                    resnet.avg_pool,
                    nn.Flatten(1,-1)
)
rgb_normalize = ()


model.eval()
model = model.to(device)
for p in model.parameters():
    p.requires_grad = False

# load dataset
transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

train_dataset = datasets.CIFAR10(root='./data/cifar10', train=True,
                                        download=True, transform=transform)
batch_size = args.batch_size
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)



# define vfop
img_size = 32
def model_normalized(x):
    return F.normalize(model(x))
if args.coords_weight == 'none':
    coords_weight = compute_coords_weight(img_size,
                                        train_dataset,
                                        model_normalized, 
                                        device = device, 
                                        n_iter = 20,
                                        test_mode = args.test_mode)
else:
    coords_weight = torch.load(args.coords_weight,map_location = device)
vfop = VectorFieldOperation(img_size,coords_weight, device = device)
coords = vfop.coords



# path
exp_path = os.path.join(experiment_name,args.exp_name)
if not os.path.exists(experiment_name):
    os.mkdir(experiment_name)
if not os.path.exists(exp_path):
    os.mkdir(exp_path)

outfile = os.path.join(exp_path,'log.txt')
with open(outfile,'w') as f:
    f.write(f'Experiment {args.exp_name}'+'\n')
with open(os.path.join(exp_path,'args.yaml'),'w') as f:
    yaml.dump(vars(args),f)

fig_path = os.path.join(exp_path,'figures')
if not os.path.exists(fig_path):
    os.mkdir(fig_path)

torch.save(coords_weight,os.path.join(exp_path,'coords_weight.pt'))
plot_coords_weight(coords_weight.detach().cpu(),os.path.join(exp_path,'coords_weight.png'))


# delta_model
n_delta = args.n_delta

delta_model = DeltaModel(vfop = vfop, n_delta = n_delta)

delta_model = delta_model.to(device)
delta_model.train()

# optimizer
optimizer = optim.Adam(delta_model.parameters(), lr=args.lr)


# hyperparameters for training
epochs = args.epochs
sigma = args.sigma
tau = args.tau
ortho_weight = args.ortho_weight
lipschitz_weight = args.lipschitz_weight
test_mode = args.test_mode

compute_lipschitz_loss = LipschitzLoss(img_size = img_size, tau=tau,device = device)

# metric tracker
metric_tracker = MetricTracker(['loss','loss_each',
                                'ortho_loss','ortho_loss_each',
                                'lipschitz_loss','lipschitz_loss_each'])
min_loss = np.inf

# functions for later use
def inner_product(a,b):
    return torch.einsum('ik,jk->ij',a,b)
def norm(a):
    return inner_product(a,a).diag()**0.5
def normalize(a):
    return a / torch.clamp(norm(a),min = 1e-8).unsqueeze(1)

# train
for epoch in range(epochs):
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        batch_size = data.shape[0]
        optimizer.zero_grad()

        #coords transformation using ode
        coords_transformed = coords.unsqueeze(3).repeat(1,1,1,n_delta).contiguous()
        t_final = (np.random.rand()*2-1) * sigma
        interval = torch.tensor([0.,t_final]).to(data.device,data.dtype)
        
        coords_transformed = odeint(lambda t,x:delta_model(x),
                                    coords_transformed,
                                    interval,
                                    adjoint_params=delta_model.parameters(),
                                    method = 'rk4',
                                    options = {'step_size' : sigma/20.})[1]
        
        
        # reshape and transform data
        coords_repeat = coords_transformed.unsqueeze(0).repeat(batch_size,1,1,1,1).permute(0,4,1,2,3).flatten(0,1)
        data_repeat = data.unsqueeze(1).repeat(1,n_delta,1,1,1).flatten(0,1)

        data_transformed = F.grid_sample(data_repeat,coords_repeat,align_corners=True)


        # get model output and compute cosine similarity
        f1 = normalize(model(data))
        f2 = normalize(model(data_transformed)).view(batch_size,n_delta,-1)

        cosine_sim = torch.einsum('ij,ikj->ik',f1,f2)
        loss_each = torch.arccos(cosine_sim.clamp(max = 1 - 1e-6)).mean(dim=0)

        # re-compute delta for regular coords and compute sequential orthonormality loss
        delta = delta_model(coords.unsqueeze(3).repeat(1,1,1,n_delta))
        ortho_loss_each = vfop.sequential_ortho_loss(delta,final=args.ortho_loss_final)
        lipschitz_loss_each = compute_lipschitz_loss(delta)
        
        
        ortho_loss = ortho_loss_each.sum() * ortho_weight
        lipschitz_loss = lipschitz_loss_each.sum() * lipschitz_weight
        loss = loss_each.sum()

        loss += ortho_loss + lipschitz_loss
        
        loss.backward()
        optimizer.step()
        # loss_each = ((1-cosine_sim)**2).sum(dim=0)
        
        # update metrics
        metric_tracker.update({
            'loss':loss.item(),
            'loss_each':loss_each.detach().cpu(),
            'ortho_loss':ortho_loss.item(),
            'ortho_loss_each':ortho_loss_each.detach().cpu(),
            'lipschitz_loss':lipschitz_loss.item(),
            'lipschitz_loss_each':lipschitz_loss_each.detach().cpu(),
        })
        
        if args.test_mode:
            if batch_idx == 3:
                break
            
    # gather metrics
    metric_dict = metric_tracker.aggregate()
    
    # record
    outstr = 'epoch {}, loss: {:.4f}, ortho_loss {:.4f}, lipschitz_loss {:.4f}\n'.format(epoch,metric_dict['loss'],metric_dict['ortho_loss'],metric_dict['lipschitz_loss'])     
    outstr += 'loss for each V:\n' + str(metric_dict['loss_each']) + '\n'
    outstr += 'ortho_loss for each V:\n' + str(metric_dict['ortho_loss_each']) + '\n'
    outstr += 'lipschitz_loss for each V:\n' + str(metric_dict['lipschitz_loss_each']) + '\n'
    outstr += '=' * 50
    
    with open(outfile,'a') as f:
        f.write(outstr +'\n')
    metric_tracker.to_pandas().to_csv(os.path.join(exp_path,'metrics.csv'))
    
    # plot delta
    delta = delta_model(coords.unsqueeze(3).repeat(1,1,1,n_delta))
    plot_delta(delta,coords,filename = os.path.join(fig_path,f'epoch{epoch}.png'))
        
    # save model if loss is at minimum
    if metric_dict['loss'] < min_loss:
        min_loss = metric_dict['loss']
        torch.save(delta_model.state_dict(),os.path.join(exp_path,'deltamodel.pt'))
        
