first train a raw resnet:
python train_resnet_augment.py --exp_name benchmark --no-augment

for learning symmetry:
python train_ode_method.py --exp_name {symmetry_exp_name} --n_delta 10 --device cuda --lr 1e-4 

for augmentation:
python train_resnet_augment.py --exp_name {augment_exp_name} --augment --delta {symmetry_exp_name} --transform_scale 0.3 --use_six
