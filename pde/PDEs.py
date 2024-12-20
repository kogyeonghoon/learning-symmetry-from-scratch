# codes from https://github.com/brandstetter-johannes/LPSDA.git

import os
import sys
import math
import numpy as np
import torch

from torch import nn
from torch.nn import functional as F

class ComputePSDiff():
    def __init__(self,N,L,device):
        self.N = N
        self.L = L
        self.device = device

        k = torch.arange(0,N,dtype = torch.float32)
        k[(N+1)//2:] = k[(N+1)//2:]-N
        
        self.k = k
        
    def __call__(self,u,order=1,dim=1,device = None):
        # compute psdiff
        device = self.device if device is None else device
        assert u.shape[dim] == self.N
        
        k = self.k.to(device)
        
        if (order %2 == 1) & (self.N % 2 == 0):
            k[self.N//2] = 0
            
        coeff_shape = [1 if i!=dim else -1 for i in range(len(u.shape))]
        coeff = torch.pow(2j *torch.pi * k/ self.L, order)
        coeff[0] = 0
        coeff[self.N//2] = 0
        coeff = coeff.view(coeff_shape)
        f = torch.fft.fft(u,dim=dim)
        df = f * coeff
        du = torch.fft.ifft(df,dim=dim)
        return du.real


class PDE(nn.Module):
    """
    Generic PDE template
    """
    def __init__(self):
        super().__init__()
        pass

    def __repr__(self):
        return "PDE"

    def pseudospectral_reconstruction(self):
        """
        A pseudospectral method template
        """
        pass


class KdV(PDE):
    """
    The Korteweg-de Vries equation:
    ut + (0.5*u**2 + uxx)x = 0
    """
    def __init__(self,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 nt_effective: int=None,
                 L: float=None,
                 lmin: float=None,
                 lmax: float=None,
                 device: torch.cuda.device = "cpu"):
        super().__init__()
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 20. if tmax is None else tmax
        # Sine frequencies for initial conditions
        self.lmin = 1 if lmin is None else lmin
        self.lmax = 3 if lmax is None else lmax
        # Number of different waves
        self.N = 10
        # Length of the spatial domain
        self.L = 128. if L is None else L
        self.grid_size = (100, 2 ** 8) if grid_size is None else grid_size
        # The effective time steps used for learning and inference
        self.nt_effective = 100 if nt_effective is None else nt_effective
        self.nt = self.grid_size[0]
        self.nx = self.grid_size[1]
        # dt and dx are slightly different due to periodicity in the spatial domain
        self.dt = self.tmax / (self.grid_size[0]-1)
        self.dx = self.L / (self.grid_size[1])
        self.device = device
        # if self.device != "cpu":
        #     # raise NotImplementedError
        self.psdiff = ComputePSDiff(self.nx,self.L,self.device)

        assert (self.grid_size[0] >= self.nt_effective)

    def __repr__(self):
        return f'KdV'
    
    def set_device(self,device):
        self.device = device
        self.psdiff.device = device

    def pseudospectral_reconstruction_batch(self, t: float, u: torch.tensor) -> torch.tensor:
        # batchwise gpu computation
        # u has shape (batch_size,nx)

        # Compute the x derivatives using the pseudo-spectral method.
        ux = self.psdiff(u)
        uxxx = self.psdiff(u, order=3)
        # Compute du/dt.
        dudt = - u*ux - uxxx
        return dudt


class KS(PDE):
    """
    The Kuramoto-Sivashinsky equation:
    ut + (0.5*u**2 + ux + uxxx)x = 0
    """
    def __init__(self,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 nt_effective: int=None,
                 L: float=None,
                 lmin: float=None,
                 lmax: float=None,
                 device: torch.cuda.device = "cpu"):
        super().__init__()
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 40. if tmax is None else tmax
        # Sine frequencies for initial conditions
        self.lmin = 1 if lmin is None else lmin
        self.lmax = 3 if lmax is None else lmax
        # Number of different waves
        self.N = 10
        # Length of the spatial domain
        self.L = 64. if L is None else L
        self.grid_size = (100, 2 ** 8) if grid_size is None else grid_size
        # The effective time steps used for learning and inference
        self.nt_effective = 100 if nt_effective is None else nt_effective
        self.nt = self.grid_size[0]
        self.nx = self.grid_size[1]
        # dt and dx are slightly different due to periodicity in the spatial domain
        self.dt = self.tmax / (self.grid_size[0]-1)
        self.dx = self.L / (self.grid_size[1])
        self.device = device
        # if self.device != "cpu":
        #     # raise NotImplementedError
        self.psdiff = ComputePSDiff(self.nx,self.L,self.device)

        # Parameters for Lie Point symmetry data augmentation
        self.time_shift = 0
        self.max_x_shift = 0.0
        self.max_velocity = 0.0

        assert (self.grid_size[0] >= self.nt_effective)

    def __repr__(self):
        return f'KS'
    
    def set_device(self,device):
        self.device = device
        self.psdiff.device = device
        
    def pseudospectral_reconstruction_batch(self, t: float, u: torch.tensor) -> torch.tensor:
        # batchwise gpu computation
        # u has shape (batch_size,nx)

        # Compute the x derivatives using the pseudo-spectral method.
        ux = self.psdiff(u,)
        uxx = self.psdiff(u, order=2)
        uxxxx = self.psdiff(u, order=4)
        # Compute du/dt.
        dudt = - u*ux - uxx - uxxxx
        return dudt
    
# def fft_diff(u,dx,dim, order = 1):

#     N = u.shape[dim]
#     L = dx * N 
#     k = torch.arange(0,N,dtype = torch.float32).to(u.device)
#     k[(N+1)//2:] = k[(N+1)//2:]-N
    
    
#     if (order %2 == 1) & (N % 2 == 0):
#         k[N//2] = 0

#     coeff = torch.pow(2j *torch.pi * k / L,order)

#     f = torch.fft.fft(u,dim=dim)
#     coeff_shape = [1 if i!=dim else -1 for i in range(len(u.shape))]
#     df = f * coeff.view(coeff_shape)
#     du = torch.fft.ifft(df,dim=dim)
#     return du.real

class Burgers(PDE):
    """
    The heat equation ut - nu * uxx = 0
    which we use to get data for the Burgers' equation via the Cole-Hopf transformation
    """
    def __init__(self,
                 nu: float=None,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 nt_effective: int=None,
                 L: float=None,
                 lmin: float=None,
                 lmax: float=None,
                 device: torch.cuda.device = "cpu"):
        super().__init__()
        # Diffusion coefficient
        self.nu = 0.01 if nu is None else nu
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 16. if tmax is None else tmax
        # Sin frequencies for initial conditions
        self.lmin = 1 if lmin is None else lmin
        self.lmax = 7 if lmax is None else lmax
        # Number of different waves
        self.N = 20
        # Length of the spatial domain
        self.L = 2 * math.pi if L is None else L
        self.grid_size = (100, 2 ** 8) if grid_size is None else grid_size
        # The effective time steps used for learning and inference
        self.nt_effective = 100 if nt_effective is None else nt_effective
        self.nt = self.grid_size[0]
        self.nx = self.grid_size[1]
        # dt and dx are slightly different due to periodicity in the spatial domain
        self.dt = self.tmax / (self.grid_size[0]-1)
        self.dx = self.L / (self.grid_size[1])
        self.device = device
        # if self.device != "cpu":
        #     # raise NotImplementedError
        self.psdiff = ComputePSDiff(self.nx,self.L,self.device)

        # Parameters for Lie Point symmetry data augmentation
        self.time_shift = 0
        self.max_x_shift = 0.0
        self.alpha = 0.0


        assert (self.grid_size[0] >= self.nt_effective)

    def __repr__(self):
        return f'Burgers'
    
    def set_device(self,device):
        self.device = device
        self.psdiff.device = device
        
    def to_burgers(self,psi: torch.Tensor, device = None):
        # cole-hopf transformation 
        # psi has shape (nt,nx)
        
        psix = self.psdiff(psi,order = 1,device = device)
        return  - (psix / psi) * (2 * self.nu)

        
    # def to_burgers(self,psi: torch.Tensor, device = None):
    #     # cole-hopf transformation 
    #     # psi has shape (nt,nx)
        
    #     dx = self.L / psi.shape[0]
    #     psix = fft_diff(psi,dx,dim=1,order=1)
    #     return  -(psix / psi) * (2 * self.nu)


    def pseudospectral_reconstruction_batch(self, t: float, u: torch.tensor) -> torch.tensor:
        # batchwise gpu computation
        # u has shape (batch_size,nx)

        # Compute the x derivatives using the pseudo-spectral method.
        uxx = self.psdiff(u, order=2)
        # Compute du/dt.
        dudt = self.nu * uxx
        return dudt
    
    
class nKdV(PDE):
    """
    The Korteweg-de Vries equation with nonlinear t:
    ut + exp(t/50)(0.5*u**2 + uxx)x = 0
    """
    def __init__(self,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 nt_effective: int=None,
                 L: float=None,
                 lmin: float=None,
                 lmax: float=None,
                 device: torch.cuda.device = "cpu"):
        super().__init__()
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 20. if tmax is None else tmax
        # Sine frequencies for initial conditions
        self.lmin = 1 if lmin is None else lmin
        self.lmax = 2 if lmax is None else lmax
        # Number of different waves
        self.N = 10
        # Length of the spatial domain
        self.L = 128. if L is None else L
        self.grid_size = (100, 2 ** 8) if grid_size is None else grid_size
        # The effective time steps used for learning and inference
        self.nt_effective = 100 if nt_effective is None else nt_effective
        self.nt = self.grid_size[0]
        self.nx = self.grid_size[1]
        # dt and dx are slightly different due to periodicity in the spatial domain
        self.dt = self.tmax / (self.grid_size[0]-1)
        self.dx = self.L / (self.grid_size[1])
        self.device = device
        # if self.device != "cpu":
        #     # raise NotImplementedError
        self.psdiff = ComputePSDiff(self.nx,self.L,self.device)

        assert (self.grid_size[0] >= self.nt_effective)

    def __repr__(self):
        return f'nKdV'
    
    def set_device(self,device):
        self.device = device
        self.psdiff.device = device

    def pseudospectral_reconstruction_batch(self, t: float, u: torch.tensor) -> torch.tensor:
        # batchwise gpu computation
        # u has shape (batch_size,nx)

        # Compute the x derivatives using the pseudo-spectral method.
        ux = self.psdiff(u)
        uxxx = self.psdiff(u, order=3)
        # Compute du/dt.
        dudt = (- u*ux - uxxx) * torch.exp(t / 50.)
        return dudt
    
class cKdV(PDE):
    """
    The cylindrical Korteweg-de Vries equation:
    ut + (0.5*u**2 + uxx)x + u /(2t+2) = 0
    """
    def __init__(self,
                 tmin: float=None,
                 tmax: float=None,
                 grid_size: list=None,
                 nt_effective: int=None,
                 L: float=None,
                 lmin: float=None,
                 lmax: float=None,
                 device: torch.cuda.device = "cpu"):
        super().__init__()
        # Start and end time of the trajectory
        self.tmin = 0 if tmin is None else tmin
        self.tmax = 20. if tmax is None else tmax
        # Sine frequencies for initial conditions
        self.lmin = 1 if lmin is None else lmin
        self.lmax = 3 if lmax is None else lmax
        # Number of different waves
        self.N = 10
        # Length of the spatial domain
        self.L = 128. if L is None else L
        self.grid_size = (100, 2 ** 8) if grid_size is None else grid_size
        # The effective time steps used for learning and inference
        self.nt_effective = 100 if nt_effective is None else nt_effective
        self.nt = self.grid_size[0]
        self.nx = self.grid_size[1]
        # dt and dx are slightly different due to periodicity in the spatial domain
        self.dt = self.tmax / (self.grid_size[0]-1)
        self.dx = self.L / (self.grid_size[1])
        self.device = device
        # if self.device != "cpu":
        #     # raise NotImplementedError
        self.psdiff = ComputePSDiff(self.nx,self.L,self.device)

        assert (self.grid_size[0] >= self.nt_effective)

    def __repr__(self):
        return f'cKdV'
    
    def set_device(self,device):
        self.device = device
        self.psdiff.device = device

    def pseudospectral_reconstruction_batch(self, t: float, u: torch.tensor) -> torch.tensor:
        # batchwise gpu computation
        # u has shape (batch_size,nx)

        # Compute the x derivatives using the pseudo-spectral method.
        ux = self.psdiff(u)
        uxxx = self.psdiff(u, order=3)
        # Compute du/dt.
        dudt = (- u*ux - uxxx - (u/(2*t+2)))
        return dudt