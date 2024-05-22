from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import math
import yaml
        
from tqdm import tqdm


def compute_coords(img_size):
    # compute coords values on coordinate grids (scaled by [-1,1])
    coords_value = torch.linspace(-1,1,img_size)
    y,x = torch.meshgrid(coords_value,coords_value)
    return torch.stack((x,y),dim=2)


def compute_coords_weight(img_size, dataset, model, device, n_iter = 20, batch_size = 32,test_mode = False):
    from kornia.filters import spatial_gradient
    from torch.autograd.functional import jvp
    
    coords = compute_coords(img_size).to(device)
    loader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=batch_size,
                                         shuffle=True, 
                                         num_workers=4)
    
    coords_weight = torch.zeros_like(coords[:,:,0])
    print("computing coords weight")
    if test_mode:
        n_iter = 1
    for i in tqdm(range(n_iter)):
        
        for batch_idx,(data,target) in enumerate(loader):
            data,target = data.to(device),target.to(device)
            batch_size = data.shape[0]
            jvp_mask_center = torch.rand(2).to(device)*2-1
            sigma = 0.1
            jvp_mask_dist = ((coords - jvp_mask_center.view(1,1,-1))**2).sum(dim=-1)**0.5
            jvp_mask = torch.exp(-0.5*(jvp_mask_dist/sigma)**2)


            jvp_input = data.unsqueeze(1).repeat(1,2,1,1,1).flatten(0,1) # shape (batch_size*2, 3, img_size, img_size)
            image_grad = spatial_gradient(data,mode = 'sobel', normalized = False)
            image_grad = torch.einsum('abcde,de->abcde',image_grad,jvp_mask)
            image_grad = image_grad.permute(0,2,1,3,4).flatten(0,1)  # shape (batch_size*2, 3, img_size, img_size)
            mask_grad = jvp(model, jvp_input,  image_grad)[1]
            mask_grad_norm = (mask_grad**2).sum()

            coords_weight += mask_grad_norm * jvp_mask
            
            if test_mode and (batch_idx == 3):
                break
        
    return coords_weight

class LipschitzLoss():
    def __init__(self,img_size,tau,device):
        self.img_size = img_size
        self.tau = tau
        
        y,x = torch.meshgrid(torch.arange(img_size),torch.arange(img_size))
        x,y = x.flatten(),y.flatten()

        flatten_idx = lambda ix,iy:ix * img_size + iy

        clip_idx = x+1<img_size
        neighbor_dx = torch.stack([flatten_idx(x[clip_idx],y[clip_idx]),flatten_idx(x[clip_idx]+1,y[clip_idx])],dim=1)
        clip_idx = y+1<img_size
        neighbor_dy = torch.stack([flatten_idx(x[clip_idx],y[clip_idx]),flatten_idx(x[clip_idx],y[clip_idx]+1)],dim=1)

        neighbor = torch.cat([neighbor_dx,neighbor_dy],dim=0)
        self.neighbor = neighbor.to(device)
    
    def __call__(self,delta):
        delta_d = delta.flatten(0,1)[self.neighbor[:,0]] - delta.flatten(0,1)[self.neighbor[:,1]]
        delta_d_norm = torch.norm(delta_d,dim=1)

        lipschitz_loss = torch.clamp(delta_d_norm-self.tau,min=0).sum(dim=0)
        
        return lipschitz_loss


class VectorFieldOperation():
    def __init__(self, img_size, coords_weight = None, device = None):
        # on coords of size img_size * img_size, computes linear operations of vector fields under weights given as coords_weight
        # by default, vector fields have size (img_size,img_size,2,) (singular) or (img_size,img_size,2,n_delta) (multiple)
        # (n_delta = number of vector fields)
        # if boundary_pixel !=0, then cut out the boundary weights of width (boundary_pixels)
        
        self.img_size = img_size
        self.coords = compute_coords(img_size).to(device)
        
        if coords_weight == None:
            coords_weight = torch.ones((img_size,img_size),dtype = torch.float32)
        coords_weight = coords_weight / coords_weight.sum()
        self.coords_weight = coords_weight.to(device)

    def get_affine_basis(self):
        # returns six affine vector fields

        x_comp = self.coords[...,0]
        y_comp = self.coords[...,1]
        one_comp = torch.ones_like(x_comp)
        zero_comp = torch.zeros_like(x_comp)

        affine_basis = [self.normalize_delta(torch.stack([one_comp,zero_comp],dim=2)),
                        self.normalize_delta(torch.stack([zero_comp,one_comp],dim=2)),
                        self.normalize_delta(torch.stack([x_comp,zero_comp],dim=2)),
                        self.normalize_delta(torch.stack([zero_comp,y_comp],dim=2)),
                        self.normalize_delta(torch.stack([y_comp,zero_comp],dim=2)),
                        self.normalize_delta(torch.stack([zero_comp,x_comp],dim=2)),
                        ]
        affine_basis = torch.stack(affine_basis,dim=3)
        
        return affine_basis

    def inner_product(self, delta1, delta2):
        # computes inner products of two (stacks of) vector fields
        
        if (len(delta1.shape) == 3) & (len(delta2.shape) == 3):
            return (torch.einsum('ijk,ijk,ij',delta1,delta2,self.coords_weight) / self.coords_weight.sum())
        
        elif (len(delta1.shape) == 4) & (len(delta2.shape) == 3):
            return (torch.einsum('ijka,ijk,ij->a',delta1,delta2,self.coords_weight) / self.coords_weight.sum())
        
        elif (len(delta1.shape) == 3) & (len(delta2.shape) == 4):
            return (torch.einsum('ijk,ijkb,ij->b',delta1,delta2,self.coords_weight) / self.coords_weight.sum())
        
        elif (len(delta1.shape) == 4) & (len(delta2.shape) == 4):
            return (torch.einsum('ijka,ijkb,ij->ab',delta1,delta2,self.coords_weight) / self.coords_weight.sum())
        
    def normalize_delta(self, delta, eps = 1e-6, detach = False):
        # normalizes vector fields
        
        if len(delta.shape) == 3:
            delta_norm = self.inner_product(delta,delta) ** 0.5
            delta_norm = torch.clamp(delta_norm,min=eps)
            
            return delta / delta_norm
        
        elif len(delta.shape) == 4:
            delta_norm = self.inner_product(delta,delta).diag() ** 0.5
            delta_norm = torch.clamp(delta_norm,min=eps)
            
            return delta / delta_norm.view(1,1,1,-1)
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
    
    def resample_coords_weight(self,new_coords):
        # sometimes vector fields are defined on non-regular coords
        # computes weights on the new coords
        
        new_coords_weight = self.coords_weight.view(1,1,self.img_size,self.img_size)
        new_coords_weight = F.grid_sample(new_coords_weight,new_coords.unsqueeze(3).permute(3,0,1,2),align_corners=True)
        
        return new_coords_weight.view(self.img_size,self.img_size)
    
    def normalize_delta_new_coords(self, delta,new_coords, eps = 1e-6):
        # normalize delta on new coords
        
        new_coords_weight = self.resample_coords_weight(new_coords)
        delta_norm = torch.einsum('ijk,ijk,ij',delta,delta,new_coords_weight) ** 0.5
        delta_norm = torch.clamp(delta_norm,min=eps)
        return delta / delta_norm
    
   
class DeltaModel(nn.Module):
    def __init__(self, vfop, n_delta = 5):
        # gets n_delta number of (irregular) coords
        # and returns n_delta number of vector fields computed on those coords
        # input shape (img_size,img_size,2,n_delta)
        # otuput shape (img_size,img_size,2,n_delta)
        # suitable for putting in ode integrator
        
        super().__init__()
        self.n_delta = n_delta
        self.vfop = vfop
        self.fc1 = nn.Linear(2, 256)
        self.fc2 = nn.Linear(256, 256)
        
        smallfc_list = []
        for _ in range(self.n_delta):
            smallfc_list.append(nn.Sequential(nn.Linear(256,32),nn.SiLU(),nn.Linear(32,2)))
        self.smallfc = nn.ModuleList(smallfc_list)
        self.silu = nn.SiLU()


    def forward(self, x,raw = False):
        new_coords = x.detach().clone()
        img_size, _,_,n_delta= x.shape
        
        x = x.permute(3,0,1,2) # shape (n_delta,img_size,img_size,2)
        x = x.reshape(n_delta*img_size*img_size,2)
        
        x = self.fc1(x)
        x = self.silu(x)
        x = self.fc2(x)
        x = self.silu(x)
        x_list = []
        for i in range(self.n_delta):
            x_list.append(self.smallfc[i](x))
        x = torch.stack(x_list,dim=2)
        x = x.view(self.n_delta,img_size,img_size,2, self.n_delta)
        x = x.permute(1,2,3,0,4)
        
        # need to get values over diagonals (since at i-th coords, only need i-th vector field)
        x = x[:,:,:,torch.arange(self.n_delta),torch.arange(self.n_delta)] # shape (img_size,img_size,2,n_delta)
        
        if raw:
            output = x
            return output

        output_list = []
        for i in range(n_delta):
            output_list.append(self.vfop.normalize_delta_new_coords(x[...,i],new_coords[...,i]))
        output = torch.stack(output_list,dim=3)

        return output
    
    def get_delta(self):
        return self.forward(self.vfop.coords.unsqueeze(3).repeat(1,1,1,self.n_delta))
    