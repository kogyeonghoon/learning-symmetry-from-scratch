import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import os
import math
import yaml
        
from tqdm import tqdm

from deltamodel import VectorFieldOperation, DeltaModel
from utils import grid_sample, plot_delta, MetricTracker

from torchdiffeq import odeint
from resnet import resnet18

experiment_name = 'exp_augment'

parser = argparse.ArgumentParser()

# basic
parser.add_argument("--augment", type = bool, default = True, action=argparse.BooleanOptionalAction)
parser.add_argument("--exp_name", type = str)
parser.add_argument("--delta", type = str, default = 'affine')
parser.add_argument("--save", type = bool, default = True, action=argparse.BooleanOptionalAction)

# optimization
parser.add_argument("--lr", type = float, default = 1e-1)
parser.add_argument("--batch_size", type = int, default = 128)
parser.add_argument("--epochs", type = int, default = 200)
parser.add_argument("--optimizer", type = str, default = 'sgd')
parser.add_argument("--scheduler", type = str, default = 'step')
parser.add_argument("--use_crop", type = bool, default = False, action=argparse.BooleanOptionalAction)
parser.add_argument("--use_six", type = bool, default = False, action=argparse.BooleanOptionalAction)

# scale
parser.add_argument("--transform_scale", type = float, default = 0.1)

# for test
parser.add_argument("--test_mode", type = bool, default = False, action=argparse.BooleanOptionalAction)


args = parser.parse_args()
device = torch.device("cuda")
# torch.cuda.set_device(device)

# define model
resnet = resnet18().to(device)

if args.use_crop:
    transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
else:
    transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
batch_size = args.batch_size
train_dataset = datasets.CIFAR10(root='./data/cifar10', train=True,
                                        download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data/cifar10', train=False,
                                        download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

# load delta
img_size = 32


if args.delta == 'affine':
    # when args.delta == affine, directly get affine basis using vfop functionality
    vfop = VectorFieldOperation(img_size,coords_weight = None, device = device)
    coords = vfop.coords
    delta = vfop.get_affine_basis()
    n_delta = 6
else:
    # when args.delta != affine, it must be path of deltamodel experiment
    with open(os.path.join(args.delta, 'args.yaml'),'r') as f:
        delta_args = yaml.load(f,Loader = yaml.FullLoader)
    delta_args = argparse.Namespace(**delta_args)
    
    n_delta = delta_args.n_delta
    coords_weight = torch.load(os.path.join(args.delta, 'coords_weight.pt')).to(dtype=torch.float32,device=device)
    vfop = VectorFieldOperation(img_size,coords_weight, device = device)
    coords = vfop.coords

    delta_model = DeltaModel(vfop = vfop,
                            n_delta = delta_args.n_delta,)
        
    delta_model.load_state_dict(torch.load(os.path.join(args.delta,'deltamodel.pt'),map_location = device))
    delta_model.to(device)
    delta = delta_model.get_delta().detach()
    
    if args.use_six:
        assert n_delta>=6
        delta = delta[...,:6]
        n_delta = 6


def resample_delta(delta_dir,coords_transformed):
    # resample delta on transformed coords using grid_sample function
    delta_transform = F.grid_sample(delta_dir.permute(3,2,0,1),coords_transformed,align_corners=True)
    return delta_transform.permute(0,2,3,1)

def ode_transform(data, coords, delta_dir):
    # ode transform data using the given delta direction
    # delta_dir shape H*W*2*n_data
    # data shape n_data*C*H*W
    # coords shape H*W*2
        
    n_data = data.shape[0]
    coords_transformed = coords.unsqueeze(0).repeat(n_data,1,1,1)
    
    interval = torch.tensor([0.,1.]).to(data.device,data.dtype)
    ode_func = lambda t,x:resample_delta(delta_dir,coords_transformed)
    coords_transformed = odeint(ode_func,coords_transformed,interval)[1]

    return F.grid_sample(data,coords_transformed,align_corners=True)


def transform_by_delta(data,delta,coords, scale = 0.1):
    u = (torch.rand((n_delta,data.shape[0]),device = data.device) * 2 - 1) * scale
    delta_dir = torch.einsum('ji,abcj->abci',u,delta)
    
    return ode_transform(data,coords,delta_dir)

# path
if not os.path.exists(experiment_name):
    os.mkdir(experiment_name)
exp_path = os.path.join(experiment_name,args.exp_name)
if not os.path.exists(exp_path):
    os.mkdir(exp_path)

outfile = os.path.join(exp_path,'log.txt')
with open(outfile,'w') as f:
    f.write(f'Experiment {args.exp_name}'+'\n')
with open(os.path.join(exp_path,'args.yaml'),'w') as f:
    yaml.dump(vars(args),f)
plot_delta(delta,coords,filename = os.path.join(exp_path,'delta.png'))


# optimizer & scheduler
criterion = nn.CrossEntropyLoss()
if args.optimizer == 'sgd':
    optimizer = optim.SGD(resnet.parameters(), weight_decay = 5e-4, lr = args.lr, momentum = 0.9, nesterov = True)
elif args.optimizer == 'adam':
    optimizer = optim.Adam(resnet.parameters(), weight_decay = 5e-4, lr = args.lr)
else:
    assert False
    
if args.scheduler == 'pluateau':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience = 3, threshold = 0.001, mode = 'max')
elif args.scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = args.epochs)
elif args.scheduler == 'step':
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,160], gamma=0.2)

# metric tracker
metric_names = ['loss','val_acc']
metric_tracker = MetricTracker(metric_names)
max_acc = 0

# hyperparameters for training
epochs = args.epochs
transform_scale = args.transform_scale

for epoch in range(epochs):
    # train
    resnet.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        batch_size = data.shape[0]
        
        if args.augment:
            data = transform_by_delta(data,delta,coords,scale = transform_scale/np.sqrt(n_delta))
        
        optimizer.zero_grad()
        outputs = resnet(data)
        
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()
        
        metric_tracker.update({'loss':loss.item()})
        if args.test_mode:
            if batch_idx == 3:
                break
    
    # evaluate
    correct = 0
    total = 0
    resnet.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            outputs = resnet(data)
            _, pred = torch.max(outputs.data, 1)
            total+= target.shape[0]
            correct += (pred == target).sum().item()
            
            if args.test_mode:
                if batch_idx == 3:
                    break
            
    # update, gather and record metrics
    acc = correct / total
    metric_tracker.update({'val_acc':acc})
    
    metric_dict = metric_tracker.aggregate()
    if args.scheduler == 'plateau':
        scheduler.step(metric_dict['loss'])
    else:
        scheduler.step()
    
    outstr = 'epoch {}, loss: {:.4f}, val_acc: {:.4f}\n'.format(epoch,metric_dict['loss'],metric_dict['val_acc'])
    with open(outfile,'a') as f:
        f.write(outstr +'\n')
    print(outstr)
    metric_tracker.to_pandas().to_csv(os.path.join(exp_path,'metrics.csv'))
    
    if metric_dict['val_acc'] > max_acc:
        if args.save:
            torch.save(resnet.state_dict(),os.path.join(exp_path,'model.pt'))