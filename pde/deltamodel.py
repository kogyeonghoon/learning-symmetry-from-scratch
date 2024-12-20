import torch
from torch import nn


class VectorFieldOperation():
    def __init__(self,):
        # compute linear operations of vector field defined on xtu space
        # by default, vector fields have size (n_xtu,3) (singular) or (n_xtu,2,n_delta) (multiple)
        # (n_delta = number of vector fields)

        pass

    def inner_product(self, delta1, delta2):
        # computes inner products of two (stacks of) vector fields
        if (len(delta1.shape) == 2) & (len(delta2.shape) == 2):
            return (torch.einsum('ij,ij',delta1,delta2)) / delta1.shape[0]
        
        elif (len(delta1.shape) == 3) & (len(delta2.shape) == 2):
            return (torch.einsum('ija,ij->a',delta1,delta2)) / delta1.shape[0]
        
        elif (len(delta1.shape) == 2) & (len(delta2.shape) == 3):
            return (torch.einsum('ij,ijb->b',delta1,delta2)) / delta1.shape[0]
        
        elif (len(delta1.shape) == 3) & (len(delta2.shape) == 3):
            return (torch.einsum('ija,ijb->ab',delta1,delta2)) / delta1.shape[0]
        
        else:
            assert False
        
                    
    def normalize_delta(self, delta, eps = 1e-8):
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
    
    def sequential_ortho_loss(self, delta, final = 'arccos'):
        # computes sequential inner product (to be used as loss)
        ip = torch.triu(self.inner_product(delta.detach(),delta),diagonal=1)
        if final == 'square':
            ip == ip**2
        elif final == 'arccos':
            ip = - torch.arccos(torch.clamp(ip.abs(),max = 1 - 1e-6)) + torch.pi/2
        else:
            assert False
        ip = ip.sum(axis=0)
        return ip
    

class DeltaModel(nn.Module):
    def __init__(self, vfop, n_delta = 5, threshold = 0.2):
        # get xtu value of shape (n_xtu,3) or (n_xtu,3,n_delta)
        # (xtu of shape (n__xtu,3,n_delta) is required to compute ODE integration)
        # and returns vector fields of shape (n_xtu,3,n_delta)
        # output vector fields are normalized (but not necessarily orthogonal)
        # threshold: for safe boundary
                
        super().__init__()
        self.n_delta = n_delta
        self.vfop = vfop
        self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.silu = nn.SiLU()
        self.threshold = threshold
        
        smallfc_list = []
        for _ in range(self.n_delta):
            smallfc_list.append(nn.Sequential(nn.Linear(256,32),self.silu,nn.Linear(32,3)))
        self.smallfc = nn.ModuleList(smallfc_list)
        

    def boundary_expand(self,xtu):
        xtu[:,0] = xtu[:,0] % 1
        x = xtu[:,0]
        n_xtu = xtu.shape[0]
        
        # add boundary values
        th = self.threshold        
        bl = x<th # left boundary
        br = x>1-th # right boundary

        n_bl = int(torch.sum(bl))
        n_br = int(torch.sum(br))
        x_bl = x[bl]
        x_br = x[br]

        one_x = torch.tensor([[1,0,0]]).to(xtu.device,torch.float32)
        xtu = torch.cat([xtu, xtu[bl] + one_x, xtu[br] - one_x],dim=0)
        return xtu, n_xtu, bl, br, n_bl, n_br, x_bl, x_br

    def boundary_aggregate(self, z, n_xtu, bl, br, n_bl, n_br, x_bl, x_br):
        # aggregate boundary vectors
        z_0 = z[:n_xtu]
        z_bl1 = z_0[bl].clone()
        z_br1 = z_0[br].clone()
        z_bl2 = z[n_xtu:n_xtu + n_bl].clone()
        z_br2 = z[n_xtu + n_bl:n_xtu + n_bl + n_br].clone()

        th = self.threshold
        w_bl = ((x_bl + th) / (2*th)).view(-1,1,1)
        z_0[bl] = z_bl1 * w_bl + z_bl2 * (1- w_bl)
        w_br = (1-(x_br-1+th) / (2*th)).view(-1,1,1)
        z_0[br] = z_br1 * w_br + z_br2 * (1- w_br)
        z = z_0
        return z

    def forward(self, xtu,eps = 1e-8):
        # if xtu has dims = 3: shape = (n_xtu,3,n_delta)
        # if xtu has dims = 2: shape = (n_xtu,3)
        
        if len(xtu.shape) == 3:
            delta_channel = True
            n_xtu, _, _ = xtu.shape
            xtu = xtu.permute(0,2,1).flatten(0,1) # to shape (n_xtu*n_delta, 3)
        elif len(xtu.shape) == 2:
            delta_channel = False
            n_xtu,_ = xtu.shape
        else:
            assert False

        xtu, n_xtu, bl, br, n_bl, n_br, x_bl, x_br = self.boundary_expand(xtu)
        
        z = self.fc1(xtu)
        z = self.silu(z)
        z = self.fc2(z)
        z = self.silu(z)
        
        z_list = []
        for i in range(self.n_delta):
            z_list.append(self.smallfc[i](z))
            
        z = torch.stack(z_list,dim=2)
        z = z.view(-1,3,self.n_delta)
        
        z = self.boundary_aggregate(z, n_xtu, bl, br, n_bl, n_br, x_bl, x_br)
        
        if delta_channel:
            z = z.view(-1,self.n_delta,3,self.n_delta)
            z = torch.diagonal(z,dim1=1,dim2=3)
               
        z_norm = self.vfop.inner_product(z,z).diag()**0.5
        output = z / torch.clamp(z_norm,min=eps)
        return output