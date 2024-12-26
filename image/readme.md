# Symmetry Learning with CIFAR-10

Our symmetry learning method with CIFAR-10 dataset requires a pretrained feature extractor. First train a CIFAR-10 classifier with raw dataset using ResNet18 by running the following code:

```
python train_resnet.py --exp_name benchmark --no-augment
```

Next, we train an MLP that parametrizes symmetry generators. Run the algorithm by the following code:

```
python train_symmetry.py --device cuda --exp_name {symmetry_experiment_name}
```

The learned symmetries can be used as augmentation to boost the classification performance. For that, run the following code:

```
python train_resnet.py --augment --exp_name {augmentation_experiment_name} --delta {symmetry_experiment_name}
```