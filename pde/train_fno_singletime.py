import argparse
import os
import pickle
import random
import gc
import torch
import numpy as np
from datetime import datetime
from torch import nn, optim
from torch.utils.data import DataLoader
from utils import *
from fno_train_helper import *
from ode_transform import no_transform, transform
from deltamodel import VectorFieldOperation, DeltaModel

def arg_parse():
    parser = argparse.ArgumentParser(description='Train an PDE solver')
    # PDE
    parser.add_argument('--device', type=str, default='cpu', help='Used device')
    parser.add_argument('--pde', type=str, default='nKdV', help='Experiment PDE: [nKdV, cKdV]')
    parser.add_argument('--train_samples', type=int, default=512, help='Number of training samples')
    parser.add_argument('--exp_name', type=str, default='tmp', help='Experiment name')

    # Model parameters
    parser.add_argument('--n_delta', type=int, default=4, help='Number of delta transformations')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--transform_batch_size', type=int, default=32, help='Transform batch size')
    parser.add_argument('--delta_exp', type=str, default='none', help='Delta experiment name')
    parser.add_argument('--num_epochs', type=int, default=40, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.4, help='Multistep LR decay')
    parser.add_argument('--sigma', nargs='+', type=float, default=[0,0,0,0], help='Sigma values')
    parser.add_argument('--n_transform', type=int, default=1, help='Number of transforms')
    parser.add_argument('--test_mode', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Test mode')
    parser.add_argument('--p_original', type=float, default=0, help='Probability of using original data')
    parser.add_argument('--pred_t', type=int, default=90, help='Prediction time step')

    # Misc
    parser.add_argument('--time_history', type=int, default=20, help="Time steps for input")
    parser.add_argument('--time_future', type=int, default=20, help="Time steps for output")
    parser.add_argument('--print_interval', type=int, default=70, help='Interval between print statements')
    return parser.parse_args()

def init_experiment(args):
    experiment_name = f'exp_fno_{args.pde}'
    exp_path, outfile = init_path(experiment_name, args.exp_name, args)
    print_out = Writer(outfile)
    print_out(f"Experiment start at: {datetime.now()}")
    return exp_path, print_out

def load_data(args):
    # Unified data paths for nKdV and cKdV
    if args.pde == 'nKdV':
        train_data_path = 'data/nKdV_train_1024_default.h5'
        valid_data_path = 'data/nKdV_valid_1024_default.h5'
        test_data_path = 'data/nKdV_test_4096_default.h5'
        pde_path = 'data/nKdV_default.pkl'
    elif args.pde == 'cKdV':
        train_data_path = 'data/cKdV_train_1024_default.h5'
        valid_data_path = 'data/cKdV_valid_1024_default.h5'
        test_data_path = 'data/cKdV_test_4096_default.h5'
        pde_path = 'data/cKdV_default.pkl'
    else:
        raise Exception("Wrong experiment")

    with open(pde_path, 'rb') as f:
        pde = pickle.load(f)

    nt = pde.nt_effective
    nx = pde.nx

    train_dataset = HDF5Dataset(train_data_path, mode='train', nt=nt, nx=nx, n_data=args.train_samples, pde=pde)
    train_loader = DataLoader(train_dataset, batch_size=args.transform_batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)

    valid_dataset = HDF5Dataset(valid_data_path, mode='valid', nt=nt, nx=nx, pde=pde)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)

    test_dataset = HDF5Dataset(test_data_path, mode='test', nt=nt, nx=nx, pde=pde)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)

    return train_loader, valid_loader, test_loader, pde

def init_delta_model(args, pde, u_scaler):
    vfop = VectorFieldOperation()
    n_delta = args.n_delta
    dt = pde.dt
    dx = pde.dx
    nt = pde.nt_effective
    nx = pde.nx
    c2 = u_scaler.c

    if args.pde == 'nKdV':
        c1 = (dt * nt) / (dx * nx)
        def lps_delta_func(xtu):
            x = xtu[:,0]
            t = xtu[:,1]
            u = xtu[:,2]
            t = t * nt * dt + 100 / 249 * 110
            ones = torch.ones_like(x)
            zeros = torch.zeros_like(x)
            delta = torch.stack([
                torch.stack([ones, zeros, zeros], dim=1),
                torch.stack([zeros, 1/torch.exp(t / 50), zeros], dim=1),
                torch.stack([50 * (torch.exp(t/50)-1) / (dx*nx), zeros, c2 * ones], dim=1),
            ], dim=2)
            delta = vfop.normalize_all(delta)
            return delta

    elif args.pde == 'cKdV':
        c1 = (dt * nt) / (dx * nx)
        def lps_delta_func(xtu):
            x = xtu[:,0]
            t = xtu[:,1]
            u = xtu[:,2]
            x = x * nx * dt
            t = t * nt * dt + 100 / 249 * 110
            u = u_scaler.inv_scale(u)

            ones = torch.ones_like(x)
            zeros = torch.zeros_like(x)
            delta = torch.stack([
                torch.stack([ones, zeros, zeros], dim=1),
                torch.stack([((t+1)**0.5) / (dx * nx), zeros, 1/(2 * ((t+1)**0.5)) * c2], dim=1),
                torch.stack([zeros, ones, zeros], dim=1),
            ], dim=2)
            delta = vfop.normalize_all(delta)
            return delta
    else:
        raise Exception("Unsupported PDE for LPS")

    delta_model = DeltaModel(vfop, n_delta).to(args.device)
    delta_experiment_name = f'exp_{args.pde}'

    if args.delta_exp not in ['none', 'lps']:
        delta_exp_path = os.path.join(delta_experiment_name, args.delta_exp)
        delta_model.load_state_dict(torch.load(os.path.join(delta_exp_path, 'deltamodel.pt'), map_location=args.device))
    elif args.delta_exp == 'lps':
        delta_model = lps_delta_func

    return delta_model

def init_model_optimizer(args, pde):
    data_creator = DataCreator(time_history=args.time_history,
                               time_future=args.time_future,
                               t_resolution=pde.nt_effective,
                               x_resolution=pde.nx).to(args.device)

    model = FNO1d(pde=pde, time_history=args.time_history, time_future=1).to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    if args.num_epochs == 20:
        milestones = [0,5,10,15]
    elif args.num_epochs == 30:
        milestones = [0,8,15,23]
    elif args.num_epochs == 40:
        milestones = [0,10,20,30]
    else:
        raise Exception("Unsupported num_epochs")

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_decay)
    return model, data_creator, optimizer, scheduler

def transform_data(train_loader, pde, args, delta_model, u_scaler, print_out):
    data_transformed_list = []
    t_eff_list = []
    reject_list = []
    nt = pde.nt_effective
    nx = pde.nx

    for i in range(args.n_transform):
        if i == 0:
            print_out(f"Applying no_transform (1/{args.n_transform})...")
            data_transformed, t_eff, reject = no_transform(train_loader, nx, nt)
            print_out("no_transform completed.")
        else:
            print_out(f"Applying transform {i+1}/{args.n_transform}...")
            data_transformed, t_eff, reject = transform(
                train_loader,
                delta_model,
                nx,
                nt,
                sigma=torch.tensor(args.sigma).to(args.device),
                u_scaler=u_scaler,
                device=args.device,
                n_delta=args.n_delta
            )
            print_out(f"Transform {i+1} completed.")
            gc.collect()
            torch.cuda.empty_cache()

        data_transformed_list.append(data_transformed)
        t_eff_list.append(t_eff)
        reject_list.append(reject)

    data_transformed = torch.stack(data_transformed_list, dim=0)
    t_eff = torch.stack(t_eff_list, dim=0)
    reject = torch.stack(reject_list, dim=0)

    augmented_dataset = AugmentedDataset(data_transformed,
                                         t_eff,
                                         reject,
                                         args.train_samples,
                                         args.n_transform,
                                         p_original=args.p_original)

    augmented_loader = DataLoader(augmented_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    return augmented_loader

def evaluate_model(args, pde, model, loader, data_creator, criterion, print_out, mode='Validation'):
    model.eval()
    losses = []
    nlosses = []
    device = args.device
    with torch.no_grad():
        for (u,) in loader:
            x = u[:,20:40,:]
            y = u[:,args.pred_t:args.pred_t+1,:]
            x, y = x.to(device), y.to(device)
            x = x.permute(0, 2, 1)
            pred = model(x)

            loss = criterion(pred.permute(0, 2, 1), y)
            nloss = loss.sum(dim=(1,2)) / (y**2).sum(dim=(1,2))
            loss = loss.sum()
            losses.append(loss.detach() / x.shape[0])
            nlosses.append(nloss.mean().detach())
    val_loss = torch.mean(torch.stack(losses)).item()
    val_nloss = torch.mean(torch.stack(nlosses)).item()
    print_out(f'{mode} loss {val_loss}, normalized loss {val_nloss}')
    return val_loss, val_nloss

def train_loop(model, optimizer, scheduler, augmented_loader, valid_loader, test_loader, args, pde, data_creator, criterion, print_out, exp_path):
    device = args.device
    if args.test_mode:
        args.n_transform = min([2, args.n_transform])

    metric_names = ['train_loss', 'val_loss', 'val_nloss', 'test_loss', 'test_nloss']
    metric_tracker = MetricTracker(metric_names)

    params = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    print_out(f'Number of parameters: {params}')

    min_val_loss = 1e10
    test_loss, test_nloss = 1e10, 1e10

    n_iters = 100 if not args.test_mode else 2

    for epoch in range(args.num_epochs):
        print_out(f"Epoch {epoch}")
        model.train()
        all_train_losses = []

        for iteration in range(n_iters):
            iteration_losses = []
            for u, t_eff in augmented_loader:
                optimizer.zero_grad()
                x = u[:,20:40,:]
                y = u[:,args.pred_t:args.pred_t+1,:]
                x, y = x.to(device), y.to(device)
                x = x.permute(0, 2, 1)
                pred = model(x)
                loss = criterion(pred.permute(0, 2, 1), y).sum()
                loss.backward()
                iteration_losses.append(loss.detach() / x.shape[0])
                optimizer.step()
            
            iteration_losses = torch.stack(iteration_losses)
            train_loss = torch.mean(iteration_losses).item()
            all_train_losses.append(train_loss)

            if (iteration % args.print_interval) == 0:
                progress = iteration / (n_iters)
                print_out(f'Training Loss (progress: {progress:.2f}): {train_loss}')

        # Evaluate on validation set
        val_loss, val_nloss = evaluate_model(args, pde, model, valid_loader, data_creator, criterion, print_out, mode='Validation')

        # Check if we should save and evaluate on test set
        if (val_loss < min_val_loss) and (epoch > args.num_epochs * 0.75):
            torch.save(model.state_dict(), os.path.join(exp_path, 'deltamodel.pt'))
            print_out("Saved model")
            min_val_loss = val_loss

            test_loss, test_nloss = evaluate_model(args, pde, model, test_loader, data_creator, criterion, print_out, mode='Test')
        else:
            test_loss, test_nloss = 1e10, 1e10
        
        metric_tracker.update({
            'train_loss': np.mean(all_train_losses),
            'val_loss': val_loss,
            'val_nloss': val_nloss,
            'test_loss': test_loss,
            'test_nloss': test_nloss,
        })
        metric_tracker.aggregate()
        metric_tracker.to_pandas().to_csv(os.path.join(exp_path, 'metrics.csv'))

        scheduler.step()
        print_out(f"current time: {datetime.now()}")

    print_out(f"Experiment end at: {datetime.now()}")

if __name__ == "__main__":
    args = arg_parse()
    exp_path, print_out = init_experiment(args)
    train_loader, valid_loader, test_loader, pde = load_data(args)

    # Scaling dict
    scale_dict = {
        'KdV': 0.3463,
        'KS': 0.2130,
        'Burgers': 20.19,
        'nKdV': 0.4345,
        'cKdV': 3.6310,
    }

    u_scale = scale_dict[args.pde]
    u_scaler = ConstantScaler(u_scale)

    delta_model = init_delta_model(args, pde, u_scaler)
    model, data_creator, optimizer, scheduler = init_model_optimizer(args, pde)

    criterion = nn.MSELoss(reduction="none")

    augmented_loader = transform_data(train_loader, pde, args, delta_model, u_scaler, print_out)

    train_loop(model, optimizer, scheduler, augmented_loader, valid_loader, test_loader, args, pde, data_creator, criterion, print_out, exp_path)
