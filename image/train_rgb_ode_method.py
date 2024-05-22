from __future__ import print_function
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

from torch.autograd.functional import jvp
from tqdm import tqdm

from utils import  MetricTracker

from torchdiffeq import odeint_adjoint as odeint
from resnet import resnet18

import faiss
import faiss.contrib.torch_utils



from openTSNE import TSNE

experiment_name = 'exp_rgb_ode_method'


parser = argparse.ArgumentParser()

# basic
parser.add_argument("--n_delta", type = int)
parser.add_argument("--exp_name", type = str)

# optimization
parser.add_argument("--lr", type = float, default = 1e-4)
parser.add_argument("--batch_size", type = int, default = 128)
parser.add_argument("--epochs", type = int, default = 50)
parser.add_argument("--scheduler", type = str, default = 'none')

# models
parser.add_argument("--resnet_name", type = str, default = 'exp_augment/benchmark/model.pt')
parser.add_argument("--normalize", type = str, default = 'none')

# scale
parser.add_argument("--ortho_weight", type = float, default = 1.)
parser.add_argument("--lipschitz_weight", type = float, default = 1.)
parser.add_argument("--sigma", type = float, default = 1.)
parser.add_argument("--tau", type = float, default = 1.)

# for test
parser.add_argument("--test_mode", type = bool, default = False, action=argparse.BooleanOptionalAction)

args = parser.parse_args()

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
device = torch.device("cuda")

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
img_size = 32

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def __call__(self, x,dim = 0):
        shape = [1]*len(x.shape)
        shape[dim] = 3
        mean = self.mean.view(shape).to(x.device)
        std = self.std.view(shape).to(x.device)
        output = (x * std) + mean
        output = torch.clamp(output, min = 0., max = 1.)
        return output

unnormalize = UnNormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


class RGBVectorFieldOperation():
    def __init__(self, z_samples, weight, device = None):
        # on coords of size img_size * img_size, computes linear operations of vector fields under weights given as coords_weight
        # by default, vector fields have size (img_size,img_size,2,) (singular) or (img_size,img_size,2,n_delta) (multiple)
        # (n_delta = number of vector fields)
        # if boundary_pixel !=0, then cut out the boundary weights of width (boundary_pixels)
        
        
        self.n_samples = z_samples.shape[0]
        self.z_samples = z_samples.to(device)
        weight = weight / weight.sum()
        self.weight = weight.to(device)
        

    def inner_product(self, delta1, delta2):
        # computes inner products of two (stacks of) vector fields
        # print(delta1.shape)
        # print(delta2.shape)
        # print(self.weight.shape)
        # print(self.weight.isnan().any())
        
        if (len(delta1.shape) == 2) & (len(delta2.shape) == 2):
            return (torch.einsum('ij,ij,i',delta1,delta2,self.weight))
        
        elif (len(delta1.shape) == 3) & (len(delta2.shape) == 2):
            return (torch.einsum('ija,ij,i->a',delta1,delta2,self.weight))
        
        elif (len(delta1.shape) == 2) & (len(delta2.shape) == 3):
            return (torch.einsum('ij,ijb,i->b',delta1,delta2,self.weight))
        
        elif (len(delta1.shape) == 3) & (len(delta2.shape) == 3):
            return (torch.einsum('ija,ijb,i->ab',delta1,delta2,self.weight))
        
    def normalize_delta(self, delta, eps = 1e-6, detach = False):
        # normalizes vector fields
        
        if len(delta.shape) == 2:
            delta_norm = self.inner_product(delta,delta) ** 0.5
            delta_norm = torch.clamp(delta_norm,min=eps)
            
            return delta / delta_norm
        
        elif len(delta.shape) == 3:
            delta_norm = self.inner_product(delta,delta).diag() ** 0.5
            delta_norm = torch.clamp(delta_norm,min=eps)
            
            return delta / delta_norm.view(1,1,-1)
        else:
            assert False
    
    def sequential_ortho_loss(self, delta,final = 'arccos'):
        # computes sequential inner product (to be used as loss)
        ip = torch.triu(self.inner_product(delta.detach(),delta),diagonal=1)
        if final == 'square':
            ip == ip**2
        elif final == 'arccos':
            ip = - torch.arccos(torch.clamp(ip.abs(),max = 1 - 1e-6)) + torch.pi/2
        else:
            assert False
        return ip.sum(axis=0)
    
class RGBDeltaModel2(nn.Module):
    def __init__(self, vfop, n_delta = 5, final_activation = 'none'):
        # get regular coords of size (img_size,img_size,2)
        # and returns vector fields of shape (img_size,img_size,2,n_delta)
        # output vector fields are normalized (but not necessarily orthogonal)
        # if raw = True in forward(), then return values before the final activation and normalization (used for debugging)
        
        super().__init__()
        self.n_delta = n_delta
        self.vfop = vfop
        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 256)
        
        smallfc_list = []
        for _ in range(self.n_delta):
            smallfc_list.append(nn.Sequential(nn.Linear(256,32),nn.SiLU(),nn.Linear(32,3)))
        self.smallfc = nn.ModuleList(smallfc_list)
        self.silu = nn.SiLU()

    def forward(self, x,eps = 1e-8):
        # x shape: (batch_size,3,)
        
        x = self.fc1(x)
        x = self.silu(x)
        x = self.fc2(x)
        x = self.silu(x)
        x_list = []
        for i in range(self.n_delta):
            x_list.append(self.smallfc[i](x))
        x = torch.stack(x_list,dim=2)
        x = x.view(-1,3,self.n_delta)
        
        x_norm = self.vfop.inner_product(x,x).diag()**0.5
        output = x / torch.clamp(x_norm,min=eps)
        return output
    
    def get_delta(self):
        return self.forward(self.vfop.z_samples)


import pickle


n_samples = 2**10

def pairwise_distance(x1,x2):
    n1 = x1.shape[0]
    n2 = x2.shape[0]
    x1 = x1.unsqueeze(1).repeat(1,n2,1)
    x2 = x2.unsqueeze(0).repeat(n1,1,1)
    return ((x1-x2)**2).sum(dim=2)**0.5


i1 = np.random.choice(len(train_dataset),size=n_samples)
i2 = np.random.choice(img_size,size=n_samples)
i3 = np.random.choice(img_size,size=n_samples)

z_samples = torch.stack([train_dataset[i1[k]][0][:,i2[k],i3[k]] for k in range(n_samples)],dim=0).to(device)
z_samples += torch.normal(0,1e-3,size = z_samples.shape,device = z_samples.device)
k=10
topk = torch.topk(pairwise_distance(z_samples,z_samples),k=k,dim=1,largest = False)[1]

index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(),3)
index.add(z_samples)

tsne = TSNE(perplexity = n_samples // 8)
tsne = tsne.fit(z_samples.cpu())
embedding = tsne.transform(z_samples.cpu())

def model_normalized(x):
    return F.normalize(model(x))



weight = torch.zeros(size=(n_samples,),dtype=torch.float32).to(z_samples.device)
n_iter = 10 if not args.test_mode else 1

for t in range(n_iter):
    for batch_idx,(data,_) in tqdm(enumerate(train_loader)):
        data = data.to(device)
        batch_size = data.shape[0]

        i = np.random.randint(n_samples)
        neighbor = topk[i]

        nearest = index.search(data.permute(0,2,3,1).flatten(0,2).contiguous(),1)[1].flatten()

        nearest_startpoint = z_samples[nearest]
        nearest_onehop = topk[nearest,torch.randint(1,k,size=nearest.shape)]
        nearest_endpoint = z_samples[nearest_onehop]
        directions = nearest_endpoint - nearest_startpoint
        if args.normalize == 'none':
            pass
        elif args.normalize == 'sqrt_dist':
            directions /= directions.norm(dim=1).clamp(min=1e-6).unsqueeze(1)**0.5
        elif args.normalize == 'dist':
            directions /= directions.norm(dim=1).clamp(min=1e-6).unsqueeze(1)
        else:
            assert False
        
        zeros = torch.zeros_like(nearest_startpoint)

        jvp_dir = torch.where(torch.isin(nearest, neighbor).unsqueeze(1).repeat(1,3),directions,zeros)
        jvp_dir = jvp_dir.view(batch_size,img_size,img_size,3).permute(0,3,1,2).contiguous()

        grad = jvp(model_normalized, data,  jvp_dir)[1]
        grad_norm = (grad ** 2).sum()

        weight[neighbor]+=grad_norm
        
vfop = RGBVectorFieldOperation(z_samples,weight)

    
def plot_delta(z_samples,delta,sigma = 0.1,filename = None, ncols = 3):
    # plot a stack of vector fields
    # need to pass the coords
    
    z_samples = z_samples.detach().cpu()
    embedding = tsne.transform(z_samples.numpy())
    raw_rgb = unnormalize(z_samples,dim=1).cpu().numpy()

    
    delta= delta.detach().cpu().numpy()
    n_delta = delta.shape[-1]
    nrows = math.ceil(n_delta/ncols)

    fig,axes = plt.subplots(nrows = nrows, ncols=ncols,figsize=(5*ncols,3 * nrows))
    
    for i in range(n_delta):
        if nrows == 1:
            plt.sca(axes[i%ncols])
        else:
            plt.sca(axes[i//ncols,i%ncols])
        endpoint = z_samples + delta[...,i] * sigma
        
        
        endpoint_embedding = tsne.transform(endpoint)
        delta_embedding = endpoint_embedding - embedding
        
        
        
        # print(embedding[:,0].shape,embedding[:,1].shape,raw_rgb.shape,)        
        plt.scatter(embedding[:,0],embedding[:,1],c = raw_rgb,alpha=0.3)
        plt.quiver(embedding[:,0],embedding[:,1],delta_embedding[:,0],delta_embedding[:,1])
    fig.tight_layout()
    if filename is None:
        plt.show(fig)
        plt.close(fig)
    else:
        plt.savefig(filename)
        plt.close(fig)


class LipschitzLoss():
    def __init__(self,vfop,k,tau,device):
        self.vfop = vfop
        self.tau = tau
        self.k = k
                
        topk = index.search(z_samples,k)[1]
        topk = topk[:,1:]
        self.topk = topk
    
    def __call__(self,delta):
        delta_d = delta[self.topk]-delta.unsqueeze(1).repeat(1,self.k-1,1,1)
        delta_d_norm = torch.norm(delta_d,dim=2)

        lipschitz_loss_each = torch.clamp(delta_d_norm-self.tau,min=0).sum(dim=(0,1)) / self.k
        return lipschitz_loss_each
    
compute_lipschitz_loss = LipschitzLoss(vfop,k=20,tau=args.tau,device=device)


test_mode = args.test_mode
ortho_weight = args.ortho_weight
lipschitz_weight = args.lipschitz_weight
sigma = args.sigma
n_delta = args.n_delta
epochs = args.epochs

# metric tracker
metric_tracker = MetricTracker(['loss','loss_each',
                                'ortho_loss','ortho_loss_each',
                                'lipschitz_loss','lipschitz_loss_each'])
min_loss = np.inf

delta_model = RGBDeltaModel2(vfop = vfop, n_delta = n_delta).to(device)

optimizer = optim.Adam(delta_model.parameters(), lr=args.lr)
if args.scheduler == 'step':
    milestones = [int(epochs*0.5)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones = milestones,
                                               gamma = 0.1)
    
for epoch in range(epochs):
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        batch_size = data.shape[0]
        
        t_final = (np.random.rand()*2-1) * sigma
        interval = torch.tensor([0.,t_final]).to(data.device,data.dtype)

        delta = delta_model(z_samples)
        data_transformed = data.unsqueeze(4).repeat(1,1,1,1,n_delta).permute(0,2,3,1,4).flatten(0,2) # shape (batch_size,img_size,img_size,3,n_delta) then flatten

        def delta_transform(t,x):
            nearest = index.search(x.permute(0,2,1).flatten(0,1).contiguous(),1)[1]
            nearest = nearest.view(batch_size*img_size*img_size,n_delta).unsqueeze(1).repeat(1,3,1)
            delta_gathered = torch.gather(delta,dim=0,index=nearest)
            
            return delta_gathered

        data_transformed = odeint(delta_transform,data_transformed,interval,adjoint_params=delta_model.parameters(),method = 'rk4',options = {'step_size' : sigma/20.})[1]

        data_transformed = data_transformed.view(batch_size, img_size,img_size,3,n_delta).permute(0,3,1,2,4)
        data_transformed = data_transformed.permute(0,4,1,2,3).flatten(0,1)

        f1 = F.normalize(model(data),dim=1)
        f2 = F.normalize(model(data_transformed),dim=1).view(batch_size,n_delta,-1)

        cosine_sim = torch.einsum('ij,ikj->ik',f1,f2)
        cosine_sim = torch.clamp(cosine_sim,max=1-1e-6,min=-1)
        
        loss_each = torch.arccos(cosine_sim).mean(dim=0)
        loss = loss_each.sum()
        ortho_loss_each = vfop.sequential_ortho_loss(delta)
        lipschitz_loss_each = compute_lipschitz_loss(delta)
        ortho_loss = ortho_loss_each.sum() * ortho_weight
        lipschitz_loss = lipschitz_loss_each.sum() * lipschitz_weight
        loss += ortho_loss + lipschitz_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
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
    plot_delta(vfop.z_samples,delta_model.get_delta(),filename = os.path.join(fig_path,f'epoch{epoch}.png'))
                
        
    # save model if loss is at minimum
    if metric_dict['loss'] < min_loss:
        min_loss = metric_dict['loss']
        torch.save(delta_model.state_dict(),os.path.join(exp_path,'deltamodel.pt'))
        
    # update scheduler
    if args.scheduler == 'step':
        scheduler.step()