from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import yaml
import pandas as pd
        
from tqdm import tqdm

def default_experiment_name():
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  # Format: YYYYMMDDHHMMSS
    return f"exp{timestamp}"

def grid_sample(image, optical):
    # an on-scratch implementation of torch.nn.functional.grid_sample
    # (unlike torch implementation) can compute second order derivatives
    # image: batch_size, n_channels, height, width
    # optical: batch_size, height, width, 2
    
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW-1);
    iy = ((iy + 1) / 2) * (IH-1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix)    * (iy_se - iy)
    ne = (ix    - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix)    * (iy    - iy_ne)
    se = (ix    - ix_nw) * (iy    - iy_nw)
    
    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW-1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH-1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW-1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH-1, out=iy_ne)
 
        torch.clamp(ix_sw, 0, IW-1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH-1, out=iy_sw)
 
        torch.clamp(ix_se, 0, IW-1, out=ix_se)
        torch.clamp(iy_se, 0, IH-1, out=iy_se)

    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) + 
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val

def plot_coords_weight(coords_weight,filename = None):
    fig = plt.figure()
    plt.imshow(coords_weight.detach().cpu())
    plt.colorbar()
    if filename is None:
        plt.show(fig)
        plt.close(fig)
    else:
        plt.savefig(filename)
        plt.close(fig)

def plot_delta(delta,coords,filename = None, ncols = 5,figsize = None,return_fig = False):
    # plot a stack of vector fields
    # need to pass the coords
    
    if len(delta.shape) == 4:
        delta = delta.reshape((-1,)+delta.shape[2:])
    
    coords_x = coords[:,:,0].detach().cpu().numpy().flatten()
    coords_y = coords[:,:,1].detach().cpu().numpy().flatten()

    delta_result = delta.detach().cpu().numpy()
    n_delta = delta.shape[-1]
    nrows = math.ceil(n_delta/ncols)
    
    figsize = (5*ncols,5 * nrows) if figsize is None else figsize
    fig,axes = plt.subplots(nrows = nrows, ncols=ncols,figsize=figsize)
    
    for i in range(n_delta):
        if nrows == 1:
            plt.sca(axes[i%ncols])
            plt.title(f'V_{i}')
        else:
            plt.sca(axes[i//ncols,i%ncols])
            plt.title(f'V_{i}')
        delta_x = delta_result[:,0,i]
        delta_y = delta_result[:,1,i]
        plt.quiver(coords_x,coords_y,delta_x,delta_y,)
    if return_fig:
        return fig,axes
    
    fig.tight_layout()
    if filename is None:
        plt.show(fig)
        plt.close(fig)
    else:
        plt.savefig(filename)
        plt.close(fig)
        
        
def plot_delta_instance(delta,coords):
    # plot a (singular) vector field
    
    coords_x = coords[:,:,0].detach().cpu().numpy().flatten()
    coords_y = coords[:,:,1].detach().cpu().numpy().flatten()

    delta_x = delta[:,0]
    delta_y = delta[:,1]
    plt.quiver(coords_x,coords_y,delta_x,delta_y,)

# experiment

def init_path(experiment_name,exp_name,args,subdirs=[]):
    os.makedirs(experiment_name,exist_ok = True)
    
    exp_path = os.path.join(experiment_name,exp_name)
    os.makedirs(exp_path,exist_ok = True)

    outfile = os.path.join(exp_path,'log.txt')
    with open(outfile,'w') as f:
        f.write(f'Experiment {exp_name}'+'\n')
    with open(os.path.join(exp_path,'args.yaml'),'w') as f:
        yaml.dump(vars(args),f)

    for subdir in subdirs:
        subdir_path=  os.path.join(exp_path,subdir)
        os.makedirs(subdir_path,exist_ok = True)

    return exp_path, outfile

class Writer():
    def __init__(self,outfile):
        self.outfile = outfile
    def __call__(self,outstr):
        with open(self.outfile,'a') as f:
            f.write(outstr +'\n')
        print(outstr)


class MetricTracker():
    # gathers metrics, take averages and record them, export to pandas
    # supports float & torch tensor (1-dim)
    
    def __init__(self,metric_names):
        self.metric_names = metric_names
        self.metric_history = {n:[] for n in self.metric_names}
        self.initialize()
        self.step = 0
        
    def initialize(self):
        self.metric_collect = {n:[] for n in self.metric_names}
        
    def update(self,new_metric):
        for n in self.metric_names:
            if n in new_metric.keys():
                self.metric_collect[n].append(new_metric[n])
            
    def aggregate(self):
        new_dict = dict()
        for n in self.metric_names:
            if len(self.metric_collect[n])==0:
                continue
            if isinstance(self.metric_collect[n][0],float):
                metric_mean = np.mean(self.metric_collect[n])
            elif type(self.metric_collect[n][0]) == torch.Tensor:
                metric_mean = torch.stack(self.metric_collect[n],dim=1).mean(dim=1).numpy()
            new_dict[n] = metric_mean
            self.metric_history[n].append(metric_mean)
        self.initialize()
        self.step += 1
        return new_dict
    
    def to_pandas(self):
        if self.step == 0:
            return pd.DataFrame()
        else:
            df_list = []
            for n in self.metric_names:
                if len(self.metric_history[n])==0:
                    continue
                if not isinstance(self.metric_history[n][0],np.ndarray):
                    df = pd.Series(self.metric_history[n],name = n)
                    df_list.append(df)
                else:
                    n_cols = len(self.metric_history[n][0])
                    df = pd.DataFrame(self.metric_history[n],columns = [n+str(i) for i in range(n_cols)])
                    df_list.append(df)
            return pd.concat(df_list,axis=1)