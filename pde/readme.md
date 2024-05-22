for data generation:
python generate_data.py --experiment=KdV --train_samples=1024 --valid_samples=1024 --test_samples=4096 --L=128 --suffix default --batch_size 4096 --device cuda
python generate_data.py --experiment=KS --train_samples=1024 --valid_samples=1024 --test_samples=4096 --L=64 --nt=500 --suffix default --batch_size 4096 --device cuda
python generate_data.py --experiment=Burgers --train_samples=1024 --valid_samples=1024 --test_samples=4096 --end_time=18. --nt=180 --suffix default --batch_size 4096 --device cuda

for learning:
python train_pde_symmetry.py --pde KdV --exp_name exp5 --scheduler step --sigma 0.4 --weight_ortho 3