import argparse
import os
import pickle
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim

from utils import *
from deltamodel import VectorFieldOperation, DeltaModel
from weno import get_calculator, compute_residual
from torchdiffeq import odeint_adjoint as odeint


# Function Definitions

def parse_args():
    parser = argparse.ArgumentParser(description='Train PDE symmetry')
    # General arguments
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--pde', type=str, default='KdV', help='PDE type [KdV, KS, Burgers]')
    parser.add_argument('--train_samples', type=int, default=1024, help='Number of training samples')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    # Task-specific arguments
    parser.add_argument('--n_delta', type=int, default=4, help='Number of deltas')
    parser.add_argument('--sigma', type=float, default=0.4, help='Transformation scale')
    parser.add_argument('--dataset', type=str, default=None, help='Custom dataset path')
    parser.add_argument('--test_mode', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Test mode')

    # Loss weighting and other parameters
    parser.add_argument('--weight_pde', type=float, default=1.0, help='Weight for PDE loss')
    parser.add_argument('--weight_ortho', type=float, default=3.0, help='Weight for orthogonality loss')
    parser.add_argument('--weight_lipschitz', type=float, default=1.0, help='Weight for Lipschitz loss')
    parser.add_argument('--weight_sobolev', type=float, default=1.0, help='Weight for Sobolev loss')
    parser.add_argument('--th', type=float, default=1e-4, help='Threshold for PDE residual value')
    parser.add_argument('--patch_size', type=int, default=64, help='Patch size')
    parser.add_argument('--n_intervals', type=int, default=4, help='Number of intervals for transformations')
    parser.add_argument('--tau', type=float, default=3.0, help='Tau for Lipschitz loss')

    return parser.parse_args()


def load_data(args):
    # Load dataset and PDE configuration
    data_path = f"data/{args.pde}_train_1024_default.h5"
    if args.dataset is not None:
        data_path = args.dataset

    pde_path = f"data/{args.pde}_default.pkl"
    with open(pde_path, 'rb') as f:
        pde = pickle.load(f)

    nt, nx = pde.nt_effective, pde.nx
    train_dataset = HDF5Dataset(data_path, mode='train', nt=nt, nx=nx, n_data=args.train_samples, pde=pde)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return train_loader, pde


def define_model(args, pde, device):
    # Define model and optimizer
    vfop = VectorFieldOperation()
    delta_model = DeltaModel(vfop, n_delta=args.n_delta).to(device)

    optimizer = optim.Adam(delta_model.parameters(), lr=args.lr)

    # Default scheduler (step-based with milestones)
    milestones = [int(args.epochs * 0.5)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    return delta_model, optimizer, scheduler


def compute_loss(delta_model, u, pde, args, epoch, device):
    scale_dict = {
        'KdV':0.3463,
        'KS':0.2130,
        'Burgers':20.19,
        'nKdV':0.4345,
        'cKdV':3.6310,
    }
    nt, nx, dt, dx = pde.nt_effective, pde.nx, pde.dt, pde.dx
    random_sign = RandomSign(args.n_delta, device)
    u_scaler = ConstantScaler(scale_dict[args.pde])
    get_patch = GetPatch(nx=nx, nt=nt, patch_size=args.patch_size)
    compute_lipschitz_loss = LipschitzLoss(tau=args.tau)
    compute_sobolev_loss = SobolevLoss(k=2, nx=nx, dx=dx)
    compute_ndiff = get_calculator(str(pde), device)

    batch_size, _, _ = u.shape
    x = torch.arange(nx, device=device, dtype=torch.float32) / nx
    t = torch.arange(nt, device=device, dtype=torch.float32) / nt
    u = u_scaler.scale(u).to(device)

    # Create xtu tensor
    xtu = torch.stack([x[None, None, :].repeat(batch_size, nt, 1),
                       t[None, :, None].repeat(batch_size, 1, nx),
                       u], dim=3)

    # Generate random patches
    patch_x = torch.randint(0, nx, size=(batch_size,))
    patch_t = torch.randint(0, nt - args.patch_size, size=(batch_size,))
    xtu_patch = get_patch(xtu, patch_x, patch_t)
    xtu = xtu.flatten(0, 2)[..., None].repeat(1, 1, args.n_delta)

    # Compute delta
    delta = delta_model(xtu)
    random_sign.set_sign()
    xtu_patch = xtu_patch.flatten(0, 2)[..., None].repeat(1, 1, args.n_delta)

    # Solve ODE for transformations
    alpha = np.random.rand() * args.sigma
    interval = torch.linspace(0., alpha, args.n_intervals + 1, device=device, dtype=torch.float32)
    func = lambda t, x: random_sign.apply(delta_model(x))
    xtu_transformed = odeint(func, xtu_patch, interval, adjoint_params=delta_model.parameters(),
                             method='rk4', options={'step_size': args.sigma / 10.})[1:]
    xtu_transformed = xtu_transformed.view(args.n_intervals * batch_size, args.patch_size, args.patch_size, 3, args.n_delta)\
        .permute(0, 4, 1, 2, 3).flatten(0, 1)

    # Compute partials and PDE residual
    partials = compute_ndiff(xtu_transformed, nt, nx, dt, dx, u_scaler)
    pde_value = compute_residual(pde, partials)
    pde_value = torch.clamp(pde_value,min=args.th)

    # Compute individual losses
    loss_pde_each = torch.log(pde_value.view(args.n_intervals * batch_size, args.n_delta, -1)).mean(dim=(0, 2))
    loss_ortho_each = VectorFieldOperation().sequential_ortho_loss(delta, final='arccos')
    loss_lipschitz_each = compute_lipschitz_loss(delta.view(batch_size, nt, nx, 3, args.n_delta),
                                                 xtu.view(batch_size, nt, nx, 3, args.n_delta).detach())
    loss_sobolev_each = compute_sobolev_loss(delta.view(batch_size, nt, nx, 3, args.n_delta))

    # Weighted losses
    loss_pde = loss_pde_each.mean() * args.weight_pde
    loss_ortho = loss_ortho_each.mean() * args.weight_ortho
    loss_lipschitz = loss_lipschitz_each.mean() * args.weight_lipschitz
    loss_sobolev = loss_sobolev_each.mean() * args.weight_sobolev if args.epochs-10 < epoch else loss_sobolev_each.mean() * 0.
    # Total loss
    total_loss = loss_pde + loss_ortho + loss_lipschitz + loss_sobolev
    if total_loss.isnan():
        raise Exception

    # Prepare metrics dictionary
    metric_dict = {
        'loss': total_loss.item(),
        'loss_pde': loss_pde.item(),
        'loss_pde_each': loss_pde_each.clone().detach().cpu(),
        'loss_ortho': loss_ortho.item(),
        'loss_ortho_each': loss_ortho_each.clone().detach().cpu(),
        'loss_lipschitz': loss_lipschitz.item(),
        'loss_lipschitz_each': loss_lipschitz_each.clone().detach().cpu(),
        'loss_sobolev': loss_sobolev.item(),
        'loss_sobolev_each': loss_sobolev_each.clone().detach().cpu(),
    }
    
    return total_loss, metric_dict


def print_metrics(epoch, metrics, print_out):
    """
    Print and log metrics for a specific epoch.

    Args:
        epoch (int): Current epoch number.
        metrics (dict): Dictionary of all metrics.
        print_out (callable): Function to log metrics.
    """
    outstr = f"Epoch {epoch}, Total Loss: {metrics['loss']:.4f}\n"
    outstr += f"  PDE Loss: {metrics['loss_pde']:.4f}\n"
    outstr += f"  PDE Loss Each: {metrics['loss_pde_each']}\n"
    outstr += f"  Ortho Loss: {metrics['loss_ortho']:.4f}\n"
    outstr += f"  Ortho Loss Each: {metrics['loss_ortho_each']}\n"
    outstr += f"  Lipschitz Loss: {metrics['loss_lipschitz']:.4f}\n"
    outstr += f"  Lipschitz Loss Each: {metrics['loss_lipschitz_each']}\n"
    outstr += f"  Sobolev Loss: {metrics['loss_sobolev']:.4f}\n"
    outstr += f"  Sobolev Loss Each: {metrics['loss_sobolev_each']}\n"
    outstr += "=" * 50
    print_out(outstr)


def train_loop(args, train_loader, delta_model, optimizer, scheduler, pde, device, exp_path, print_out):
    metric_names = ['loss', 'loss_pde', 'loss_ortho', 'loss_lipschitz', 'loss_sobolev'] + ['loss_pde_each', 'loss_ortho_each', 'loss_lipschitz_each', 'loss_sobolev_each']
    metric_tracker = MetricTracker(metric_names)
    min_loss = float('inf')

    for epoch in range(args.epochs):
        for batch_idx, (u,) in tqdm(enumerate(train_loader)):
            loss, metrics = compute_loss(delta_model, u, pde, args, epoch, device)
            loss.backward()
            for name, param in delta_model.named_parameters():
                if torch.isnan(param.grad).any():
                    raise Exception
            optimizer.step()
            optimizer.zero_grad()
            metric_tracker.update(metrics)
            if args.test_mode and batch_idx == 2:
                break

        # Aggregate metrics at the end of the epoch
        epoch_metrics = metric_tracker.aggregate()
        print_metrics(epoch, epoch_metrics, print_out)

        # Save metrics to CSV
        metrics_csv_path = os.path.join(exp_path, 'metrics.csv')
        metric_tracker.to_pandas().to_csv(metrics_csv_path)

        # Save model if loss is at minimum
        if epoch_metrics['loss'] < min_loss:
            min_loss = epoch_metrics['loss']
            model_save_path = os.path.join(exp_path, 'deltamodel.pt')
            torch.save(delta_model.state_dict(), model_save_path)
        torch.save(delta_model.state_dict(), os.path.join(exp_path,'checkpoints' f'epoch{epoch}.pt'))
        # Update scheduler
        scheduler.step()

    print_out(f"Experiment end at: {datetime.now()}")


# Main Function
def main():
    args = parse_args()
    args.exp_name = default_experiment_name() if args.exp_name is None else args.exp_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize paths
    experiment_name = f'exp_{args.pde}'
    exp_path, outfile = init_path(experiment_name, args.exp_name, args, subdirs=['checkpoints'])
    print_out = Writer(outfile)
    print_out(f"Experiment start at: {datetime.now()}")

    # Load data
    train_loader, pde = load_data(args)

    # Define model
    delta_model, optimizer, scheduler = define_model(args, pde, device)

    # Train
    train_loop(args, train_loader, delta_model, optimizer, scheduler, pde, device, exp_path, print_out)


if __name__ == '__main__':
    main()





