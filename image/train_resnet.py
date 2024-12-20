import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import yaml
import numpy as np
from tqdm import tqdm
from torchdiffeq import odeint
from deltamodel import VectorFieldOperation, DeltaModel
from utils import *
from resnet import resnet18


def load_data(batch_size, use_crop):
    """Load CIFAR-10 dataset with optional cropping transform."""
    if use_crop:
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

    train_dataset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    # basic
    parser.add_argument("--augment", type=bool, default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--delta", type=str, default='affine')
    parser.add_argument("--save", type=bool, default=True, action=argparse.BooleanOptionalAction)

    # optimization
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--optimizer", type=str, default='sgd')
    parser.add_argument("--scheduler", type=str, default='step')
    parser.add_argument("--use_crop", type=bool, default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use_six", type=bool, default=True, action=argparse.BooleanOptionalAction)

    # scale
    parser.add_argument("--transform_scale", type=float, default=0.1)

    # for test
    parser.add_argument("--test_mode", type=bool, default=False, action=argparse.BooleanOptionalAction)

    return parser.parse_args()

def load_delta(args, device):
    """Load delta based on the type specified in args (either 'affine' or a model path)."""
    img_size = 32

    if args.delta == 'affine':
        # When args.delta == 'affine', get affine basis directly
        vfop = VectorFieldOperation(img_size, coords_weight=None, device=device)
        coords = vfop.coords
        delta = vfop.get_affine_basis()
        n_delta = 6
    else:
        # Load delta model from specified path
        with open(os.path.join(args.delta, 'args.yaml'), 'r') as f:
            delta_args = yaml.load(f, Loader=yaml.FullLoader)
        delta_args = argparse.Namespace(**delta_args)

        n_delta = delta_args.n_delta
        coords_weight = torch.load(os.path.join(args.delta, 'coords_weight.pt')).to(dtype=torch.float32, device=device)
        vfop = VectorFieldOperation(img_size, coords_weight, device=device)
        coords = vfop.coords

        delta_model = DeltaModel(vfop=vfop, n_delta=delta_args.n_delta)
        delta_model.load_state_dict(torch.load(os.path.join(args.delta, 'deltamodel.pt'), map_location=device))
        delta_model.to(device)
        delta = delta_model.get_delta().detach()

        if args.use_six:
            assert n_delta >= 6
            delta = delta[..., :6]
            n_delta = 6

    return delta, coords, n_delta

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
    ode_func = lambda t,x:resample_delta(delta_dir,x)
    coords_transformed = odeint(ode_func,coords_transformed,interval)[1]

    return F.grid_sample(data,coords_transformed,align_corners=True)


def transform_by_delta(data,delta,coords, scale = 0.1):
    u = (torch.rand((n_delta,data.shape[0]),device = data.device) * 2 - 1) * scale
    delta_dir = torch.einsum('ji,abcj->abci',u,delta)
    
    return ode_transform(data,coords,delta_dir)


def get_optimizer(model, args):
    """Initialize optimizer and scheduler based on provided arguments."""
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), weight_decay=5e-4, lr=args.lr, momentum=0.9, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), weight_decay=5e-4, lr=args.lr)
    else:
        raise ValueError("Unsupported optimizer type")

    if args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, threshold=0.001, mode='max')
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.epochs)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    else:
        raise ValueError("Unsupported scheduler type")

    return optimizer, scheduler


def train_loop(resnet, train_loader, test_loader, criterion, optimizer, scheduler, device, args, delta, coords, n_delta, transform_scale, exp_path, print_out):
    """Main training loop for the model."""
    metric_tracker = MetricTracker(['loss', 'val_acc'])
    max_acc = 0

    for epoch in range(args.epochs):
        # Training phase
        resnet.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            if args.augment:
                data = transform_by_delta(data, delta, coords, scale=transform_scale/np.sqrt(n_delta))

            optimizer.zero_grad()
            outputs = resnet(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            metric_tracker.update({'loss': loss.item()})
            if args.test_mode and batch_idx == 3:
                break

        # Evaluation phase
        correct = 0
        total = 0
        resnet.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                outputs = resnet(data)
                _, pred = torch.max(outputs.data, 1)
                total += target.shape[0]
                correct += (pred == target).sum().item()
                if args.test_mode and batch_idx == 3:
                    break

        acc = correct / total
        metric_tracker.update({'val_acc': acc})
        metric_dict = metric_tracker.aggregate()

        if args.scheduler == 'plateau':
            scheduler.step(metric_dict['loss'])
        else:
            scheduler.step()

        outstr = f'epoch {epoch}, loss: {metric_dict["loss"]:.4f}, val_acc: {metric_dict["val_acc"]:.4f}\n'
        print_out(outstr)
        metric_tracker.to_pandas().to_csv(os.path.join(exp_path, 'metrics.csv'))

        # Save best model
        if acc > max_acc and args.save:
            torch.save(resnet.state_dict(), os.path.join(exp_path, 'model.pt'))
            max_acc = acc


# Main script
if __name__ == "__main__":
    args = parse_args()
    args.exp_name = default_experiment_name() if args.exp_name is None else args.exp_name
    exp_path, outfile = init_path('experiment-resnet_training', args.exp_name, args, subdirs=['figures'])
    print_out = Writer(outfile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data and model
    train_loader, test_loader = load_data(args.batch_size, args.use_crop)
    resnet = resnet18().to(device)

    # Load delta
    delta, coords, n_delta = load_delta(args, device)

    # Set up optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer(resnet, args)

    # Run training loop
    train_loop(resnet, train_loader, test_loader, criterion, optimizer, scheduler, device, args, delta, coords, n_delta, args.transform_scale, exp_path, print_out)
