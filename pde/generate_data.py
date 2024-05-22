# codes from https://github.com/brandstetter-johannes/LPSDA.git

import argparse
import os
import sys
import math
import numpy as np
import torch
import h5py
import random
import pickle

from typing import Tuple
from copy import copy
from datetime import datetime
from PDEs import PDE, KdV, KS, Burgers, nKdV, cKdV
from torchdiffeq import odeint


def check_files(pde: PDE, modes: dict) -> None:
    """
    Check if data files exist and replace them if wanted.
    Args:
        pde (PDE): pde at hand [KS, KdV, Burgers]
        modes (dict): mode ([train, valid, test]), replace, num_samples, training suffix
    Returns:
            None
    """
    for mode, replace, num_samples, suffix in modes:
        save_name = "data/" + "_".join([str(pde), mode])
        save_name = save_name + "_" + str(num_samples)
        if suffix:
            save_name = save_name + "_" + suffix
        if (replace == True):
            if os.path.exists(f'{save_name}.h5'):
                os.remove(f'{save_name}.h5')
                print(f'File {save_name}.h5 is deleted.')
            else:
                print(f'No file {save_name}.h5 exists yet.')
        else:
            print(f'File {save_name}.h5 is kept.')

def check_directory() -> None:
    """
    Check if data and log directories exist, and create otherwise.
    """
    if os.path.exists(f'data'):
        print(f'Data directory exists and will be written to.')
    else:
        os.mkdir(f'data')
        print(f'Data directory created.')

    if not os.path.exists(f'data/log'):
        os.mkdir(f'data/log')

def initial_conditions(A: np.ndarray, phi: np.ndarray, l: np.ndarray, L: float):
    """
    Return initial conditions based on initial parameters.
    Args:
        A (np.ndarray): amplitude of different sine waves
        phi (np.ndarray): phase shift of different sine waves
        l (np.ndarray): frequency of different sine waves
        L (float): length of the spatial domain
    Returns:
        None
    """
    def fnc(x):
        u = np.sum(A * np.sin(2 * np.pi * l * x / L + phi), -1)
        return u
    return fnc

def params(pde: PDE, batch_size: int, device: torch.cuda.device="cpu") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get initial parameters for KdV, KS, and Burgers' equation.
    Args:
        pde (PDE): pde at hand [KS, KdV, Burgers]
        batch_size (int): batch size
        device: device (cpu/gpu)
    Returns:
        np.ndarray: amplitude of different sin waves
        np.ndarray: phase shift
        np.ndarray: space dependent frequency
    """
    A = np.random.rand(1, pde.N) - 0.5
    phi = 2.0 * np.pi * np.random.rand(1, pde.N)
    l = np.random.randint(pde.lmin, pde.lmax, (1, pde.N))
    return A, phi, l

def inv_cole_hopf(psi0: np.ndarray, scale: float = 10.) -> np.ndarray:
    """
    Inverse Cole-Hopf transformation to obtain Heat equation out of initial conditions of Burgers' equation.
    Args:
        psi0 (np.ndarray): Burgers' equation (at arbitrary timestep) which gets transformed into Heat equation
        scale (float): scaling factor for transformation
    Returns:
        np.ndarray: transformed Heat equation
    """
    psi0 = psi0 - np.amin(psi0)
    psi0 = scale * 2 * ((psi0 / np.amax(psi0)) - 0.5)
    psi0 = np.exp(psi0)
    return psi0

def generate_trajectories(pde: PDE,
                          mode: str,
                          num_samples: int,
                          suffix: str,
                          batch_size: int,
                          device: torch.cuda.device="cpu"
                          ) -> None:
    """
    Generate data trajectories for KdV, KS equation on periodic spatial domains.
    Args:
        pde (PDE): pde at hand [KS, KdV, Burgers]
        mode (str): [train, valid, test]
        num_samples (int): how many trajectories do we create
        suffix (str): naming suffix for special trajectories
        batch_size (int): batch size
        device: device (cpu/gpu)
    Returns:
        None
    """

    num_batches = int(np.ceil(num_samples / batch_size))

    pde_string = str(pde)
    print(f'Equation: {pde_string}')
    print(f'Mode: {mode}')
    print(f'Number of samples: {num_samples}')

    sys.stdout.flush()

    save_name = "data/" + "_".join([pde_string, mode])
    save_name = save_name + "_" + str(num_samples)
    if suffix:
        save_name = save_name + '_' + suffix
    h5f = h5py.File("".join([save_name, '.h5']), 'a')
    dataset = h5f.create_group(mode)

    h5f_u = {}

    # Tolerance of the solver
    tol = 1e-9
    
    
    nt = pde.grid_size[0]
    nx = pde.grid_size[1]
    
    # The field u, the coordinations (xcoord, tcoord) and dx, dt are saved
    # Only nt_effective time steps of each trajectories are saved
    h5f_u = dataset.create_dataset(f'pde_{pde.nt_effective}-{nx}', (num_samples, pde.nt_effective, nx), dtype=float)

    for batch_idx in range(num_batches):
        n_data = min((batch_idx+1) * batch_size,num_samples) - batch_idx * batch_size
        if n_data == 0:
            continue
        u0_list = []
        for trial in range(5):
            try:
                for j in range(n_data):
                    
                    T = pde.tmax
                    L = pde.L

                    t = np.linspace(pde.tmin, T, nt)
                    x = np.linspace(0, (1 - 1.0 / nx) * L, nx)

                    # Parameters for initial conditions
                    A, omega, l = params(pde, batch_size, device=device)

                    # Initial condition of the equation at end
                    u0 = initial_conditions(A, omega, l, L)(x[:, None])

                    # We use the initial condition of Burgers' equation and inverse Cole-Hopf transform it into the Heat equation
                    if pde_string == 'Burgers':
                        u0 = inv_cole_hopf(u0)

                    u0_list.append(u0)
                    
            
                spectral_method = pde.pseudospectral_reconstruction_batch
                        
                u0 = torch.tensor(np.stack(u0_list,axis=0)).to(device)
                t = torch.tensor(t).to(device)

                solved_trajectory = odeint(func=spectral_method,
                                            t=t,
                                            y0=u0,
                                            method=args.solver,
                                            atol=tol,
                                            rtol=tol)
                break
            except AssertionError:
                print(f'An error occured - possibly an underflow. re-running {trial}/5')
        
        sol = solved_trajectory.permute(1,0,2)[:,-pde.nt_effective:]
        # if pde_string == 'Burgers':
        #     sol = pde.to_burgers(sol)
            
        sol = sol.cpu()
        h5f_u[batch_idx * batch_size:batch_idx * batch_size+n_data, :, :] = sol
        
        print("Solved indices: {:d} : {:d}".format(batch_idx * batch_size, (batch_idx + 1) * batch_size - 1))
        print("Solved batches: {:d} of {:d}".format(batch_idx + 1, num_batches))
        sys.stdout.flush()

    print()
    print("Data saved")
    print()
    print()
    h5f.close()



def generate_data(experiment: str,
                  starting_time : float,
                  end_time: float,
                  L: float,
                  nx: int,
                  nt: int,
                  nt_effective: int,
                  num_samples_train: int,
                  num_samples_valid: int,
                  num_samples_test: int,
                  batch_size: int=1,
                  device: torch.cuda.device="cpu",
                  nu: float=0.01) -> None:
    """
    Generate data for KdV, KS equation on periodic spatial domains.
    Args:
        experiment (str): pde at hand [KS, KdV, Heat]
        starting_time (float): starting time of PDE solving
        end_time (float): end time of PDE solving
        L (float): length of the spatial domain
        nx (int): spatial resolution
        nt (int): temporal resolution
        num_samples_train (int): number of trajectories created for training
        num_samples_valid (int): number of trajectories created for validation
        num_samples_test (int): number of trajectories created for testing
        batch_size (int): batch size
        device: device (cpu/gpu)
    Returns:
        None
    """
    print(f'Generating data')
    dateTimeObj = datetime.now()
    timestring = f'{dateTimeObj.date().month}{dateTimeObj.date().day}{dateTimeObj.time().hour}{dateTimeObj.time().minute}'
    if args.log:
        logfile = f'data/log/{experiment}_time{timestring}.csv'
        print(f'Writing to log file {logfile}')
        sys.stdout = open(logfile, 'w')

    # Create instances of PDE
    if experiment == 'KdV':
        pde = KdV(tmin=starting_time,
                  tmax=end_time,
                  grid_size=(nt, nx),
                  nt_effective=nt_effective,
                  L=L,
                  device=device)
    elif experiment == 'KS':
        pde = KS(tmin=starting_time,
                 tmax=end_time,
                 grid_size=(nt, nx),
                 nt_effective=nt_effective,
                 L=L,
                 device=device)

    elif experiment == 'Burgers':
        # Heat equation is generated; afterwards trajectories are transformed via Cole-Hopf transformation.
        # L is not set for Burgers equation, since it is very sensitive. Default value is 2*math.pi.
        pde = Burgers(tmin=starting_time,
                 tmax=end_time,
                 grid_size=(nt, nx),
                 nt_effective=nt_effective,
                 nu=nu,
                 device=device)
    elif experiment == 'nKdV':
        pde = nKdV(tmin=starting_time,
                  tmax=end_time,
                  grid_size=(nt, nx),
                  nt_effective=nt_effective,
                  L=L,
                  device=device)
    elif experiment == 'cKdV':
        pde = cKdV(tmin=starting_time,
                  tmax=end_time,
                  grid_size=(nt, nx),
                  nt_effective=nt_effective,
                  L=L,
                  device=device)
    else:
        raise Exception("Wrong experiment")

    # Check if train, valid and test files already exist and replace if wanted
    files = {("train", num_samples_train > 0, num_samples_train, args.suffix),
             ("valid", num_samples_valid > 0, num_samples_valid, args.suffix),
             ("test", num_samples_test > 0, num_samples_test, args.suffix)}
    check_files(pde, files)
    
    # save pde object
    with open(os.path.join('data',f'{str(pde)}_{args.suffix}.pkl'),'wb') as f:
        pickle.dump(pde,f)

    # Obtain trajectories for different modes
    for mode, _, num_samples, suffix in files:
        if num_samples > 0:
            generate_trajectories(pde=pde,
                                mode=mode,
                                num_samples=num_samples,
                                suffix=suffix,
                                batch_size=batch_size,
                                device=device,)


def main(args: argparse) -> None:
    """
    Main method for data generation.
    """
    check_directory()
    generate_data(experiment=args.experiment,
                  starting_time=0.0,
                  end_time=args.end_time,
                  L=args.L,
                  nt=args.nt,
                  nt_effective=args.nt_effective,
                  nx=args.nx,
                  num_samples_train=args.train_samples,
                  num_samples_valid=args.valid_samples,
                  num_samples_test=args.test_samples,
                  batch_size=args.batch_size,
                  device=args.device,
                  nu=args.nu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generating PDE data')
    parser.add_argument('--experiment', type=str, default='KdV',
                        help='Experiment for which data should create for: [KdV, KS, Burgers, nKdV, cKdV]')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Used device')
    parser.add_argument('--end_time', type=float, default=100.,
                        help='How long do we want to simulate')
    parser.add_argument('--nt', type=int, default=250,
                        help='Time steps used for solving')
    parser.add_argument('--nt_effective', type=int, default=140,
                        help='Solved timesteps used for training')
    parser.add_argument('--nx', type=int, default=256,
                        help='Spatial resolution')
    parser.add_argument('--L', type=float, default=128.,
                        help='Length for which we want to solve the PDE')
    parser.add_argument('--train_samples', type=int, default=0,
                        help='Samples in the training dataset')
    parser.add_argument('--valid_samples', type=int, default=0,
                        help='Samples in the validation dataset')
    parser.add_argument('--test_samples', type=int, default=0,
                        help='Samples in the test dataset')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size used for creating training, val, and test dataset')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix for additional datasets')
    parser.add_argument('--log', type=eval, default=False,
                        help='pip the output to log file')
    parser.add_argument('--solver', type=str, default='dopri5',
                        help='ode solver')
    parser.add_argument('--nu', type=float, default=0.01,
                        help='nu in burgers')

    args = parser.parse_args()
    main(args)