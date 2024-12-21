for data generation:

python generate_data.py --experiment=KdV --train_samples=1024 --valid_samples=1024 --test_samples=4096 --L=128 --suffix default --batch_size 4096 --device cuda
python generate_data.py --experiment=KS --train_samples=1024 --valid_samples=1024 --test_samples=4096 --L=64 --nt=500 --suffix default --batch_size 4096 --device cuda
python generate_data.py --experiment=Burgers --train_samples=1024 --valid_samples=1024 --test_samples=4096 --end_time=18. --nt=180 --suffix default --batch_size 4096 --device cuda
python generate_data.py --experiment=nKdV --train_samples=1024 --valid_samples=1024 --test_samples=4096 --L=128 --suffix default --batch_size 4096 --device cuda
python generate_data.py --experiment=cKdV --train_samples=1024 --valid_samples=1024 --test_samples=4096 --L=128 --suffix default --batch_size 4096 --device cuda

for learning:

python train_pde_symmetry.py --pde KdV --exp_name {symmetry_exp_name} --sigma 0.4 --weight_ortho 3

for augmentation:

python train_fno.py --device cuda --pde KdV --train_samples 512 --n_delta 4 --transform_batch_size 32 --delta_exp {symmetry_exp_name} --sigma 0.1 0.1 0.1 0 --n_transform 16 --p_original 0.5
