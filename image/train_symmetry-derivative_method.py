import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import numpy as np
from datetime import datetime
from tqdm import tqdm

from deltamodel import VectorFieldOperation, DeltaModel, compute_coords_weight, LipschitzLoss
from utils import *
from torchdiffeq import odeint_adjoint as odeint
from resnet import resnet18
from torch.autograd.functional import jvp

def parse_args():
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--n_delta", type=int, default = 10)
    parser.add_argument("--exp_name", type=str, default = None)

    # optimization
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)

    # models
    parser.add_argument("--resnet_name", type=str, default='experiment-resnet_training/benchmark/model.pt')

    # scale
    parser.add_argument("--ortho_weight", type=float, default=10.)
    parser.add_argument("--ortho_loss_final", type=str, default='arccos')
    parser.add_argument("--lipschitz_weight", type=float, default=10.)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--device", type=str, default='cpu')
    
    # for test
    parser.add_argument("--test_mode", type=bool, default=False, action=argparse.BooleanOptionalAction)

    return parser.parse_args()



def create_model(device, resnet_name):
    resnet = resnet18()
    resnet.load_state_dict(torch.load(resnet_name, map_location=device))
    model = nn.Sequential(
        resnet.conv1,
        resnet.conv2_x,
        resnet.conv3_x,
        resnet.conv4_x,
        resnet.conv5_x,
        resnet.avg_pool,
        nn.Flatten(1, -1)
    )
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_data(batch_size):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_dataset = datasets.CIFAR10(root='./data/cifar10', train=True,
                                     download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=4)
    return train_loader


def train_loop(model, delta_model, optimizer, train_loader, coords, vfop, img_size, device, args, exp_path, print_out):
    compute_lipschitz_loss = LipschitzLoss(img_size=img_size, tau=args.tau, device=device)
    metric_tracker = MetricTracker(['loss', 'loss_each', 'ortho_loss', 'ortho_loss_each',
                                    'lipschitz_loss', 'lipschitz_loss_each'])
    min_loss = np.inf

    for epoch in range(args.epochs):
        for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
            data, target = data.to(device), target.to(device)
            batch_size = data.shape[0]
            optimizer.zero_grad()
            n_delta = args.n_delta
            sigma = args.sigma
            
            delta = delta_model(coords.unsqueeze(3).repeat(1,1,1,n_delta))

            # random perturbation
            u = np.random.normal(0,1)
            u_delta = u * delta.unsqueeze(0).repeat(batch_size,1,1,1,1).permute(0,4,1,2,3) # shape (batch_size, n_delta, img_size, img_size, 2)

            # coords transformation (under small perturbation by u_delta)
            coords_transformed = coords.view((1,1,img_size,img_size,2)).repeat(batch_size,n_delta,1,1,1) # shape (batch_size, n_delta, img_size, img_size, 2)
            coords_transformed = coords_transformed + u_delta * sigma

            # function put inside jvp
            def delta_transform(coords_transformed):
                data_repeat = data.unsqueeze(1).repeat(1,n_delta,1,1,1).flatten(0,1)
                data_transformed = grid_sample(data_repeat,coords_transformed.flatten(0,1))
                output = model(data_transformed).view(batch_size,n_delta,-1)
                
                # normalize
                output_norm = torch.clamp(torch.einsum('ijk,ijk->ij',output,output)**0.5, min=1e-6).unsqueeze(2)
                return output / output_norm

            # compute loss1 = norm of gradient
            delta_direction = delta.unsqueeze(0).repeat(batch_size,1,1,1,1).permute(0,4,1,2,3)  # shape (batch_size, n_delta, img_size, img_size, 2)
            grad = jvp(delta_transform, coords_transformed,  delta_direction, create_graph=True)[1]
            
            grad_norm = torch.einsum('ijk,ijk->ij',grad,grad)
            loss_each = grad_norm.sum(dim=0).log()

            # re-compute delta for regular coords and compute sequential orthonormality loss
            delta = delta_model(coords.unsqueeze(3).repeat(1, 1, 1, args.n_delta))
            ortho_loss_each = vfop.sequential_ortho_loss(delta, final=args.ortho_loss_final)
            lipschitz_loss_each = compute_lipschitz_loss(delta)

            ortho_loss = ortho_loss_each.sum() * args.ortho_weight
            lipschitz_loss = lipschitz_loss_each.sum() * args.lipschitz_weight
            loss = loss_each.sum() + ortho_loss + lipschitz_loss

            loss.backward()
            optimizer.step()

            # update metrics
            metric_tracker.update({
                'loss': loss.item(),
                'loss_each': loss_each.detach().cpu(),
                'ortho_loss': ortho_loss.item(),
                'ortho_loss_each': ortho_loss_each.detach().cpu(),
                'lipschitz_loss': lipschitz_loss.item(),
                'lipschitz_loss_each': lipschitz_loss_each.detach().cpu(),
            })

            if args.test_mode and batch_idx == 3:
                break
        min_loss = save_metrics_and_model(epoch, exp_path, print_out, metric_tracker, delta_model, coords, args, vfop, min_loss)

        


def save_metrics_and_model(epoch, exp_path, print_out, metric_tracker, delta_model, coords, args, vfop, min_loss):
    # gather metrics
    metric_dict = metric_tracker.aggregate()

    # record
    outstr = 'epoch {}, loss: {:.4f}, ortho_loss {:.4f}, lipschitz_loss {:.4f}\n'.format(
        epoch, metric_dict['loss'], metric_dict['ortho_loss'], metric_dict['lipschitz_loss'])
    outstr += 'loss for each V:\n' + str(metric_dict['loss_each']) + '\n'
    outstr += 'ortho_loss for each V:\n' + str(metric_dict['ortho_loss_each']) + '\n'
    outstr += 'lipschitz_loss for each V:\n' + str(metric_dict['lipschitz_loss_each']) + '\n'
    outstr += '=' * 50

    
    
    print_out(outstr + '\n')
    metric_tracker.to_pandas().to_csv(os.path.join(exp_path, 'metrics.csv'))

    delta = delta_model(coords.unsqueeze(3).repeat(1, 1, 1, args.n_delta))
    plot_delta(delta, coords, filename=os.path.join(exp_path, 'figures', f'epoch{epoch}.png'))

    # save model if loss is at minimum
    if metric_dict['loss'] < min_loss:
        torch.save(delta_model.state_dict(), os.path.join(exp_path, 'deltamodel.pt'))
        min_loss = metric_dict['loss']
    return min_loss


def main():
    args = parse_args()
    args.exp_name = default_experiment_name() if args.exp_name is None else args.exp_name
    exp_path, outfile = init_path('experiment-symmetry_learning', args.exp_name, args, subdirs=['figures'])
    print_out = Writer(outfile)
    
    device = args.device
    model = create_model(device, args.resnet_name)
    train_loader = load_data(args.batch_size)

    img_size = 32
    coords_weight = compute_coords_weight(img_size, train_loader.dataset, model, device=device, n_iter=20, test_mode=args.test_mode)

    vfop = VectorFieldOperation(img_size, coords_weight, device=device)
    coords = vfop.coords

    delta_model = DeltaModel(vfop=vfop, n_delta=args.n_delta).to(device).train()
    optimizer = optim.Adam(delta_model.parameters(), lr=args.lr)

    # Start training loop
    train_loop(model, delta_model, optimizer, train_loader, coords, vfop, img_size, device, args, exp_path, print_out)
    

if __name__ == "__main__":
    main()
    