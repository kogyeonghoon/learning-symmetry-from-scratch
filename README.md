# Learning Infinitesimal Generators of Continuous Symmetries from Data

This repository contains the official Python implementation of the algorithm described in the paper:  
> **"Learning Infinitesimal Generators of Continuous Symmetries from Data"**  
> *Gyeonghoon Ko, Hyunsu Kim, Juho Lee*  
> Presented at NeurIPS 2024.  

---

## Introduction

Symmetry plays a crucial role in understanding the structure and dynamics of systems across scientific disciplines. Leveraging symmetries in data can significantly enhance model efficiency, generalization, and interpretability. However, identifying and utilizing the specific symmetries present in real-world datasets is often a challenging task, especially when the symmetries extend beyond linear or affine transformations.

This repository implements a novel algorithm to learn continuous symmetries from data, using one-parameter groups defined by vector fields called infinitesimal generators. Unlike previous methods that rely on predefined Lie groups, this approach operates with minimal inductive bias and is capable of discovering both affine and non-affine symmetries.

Key features of the algorithm include:  
- A differentiable validity score that quantifies whether transformed data preserves task-specific invariances.  
- The use of Neural Ordinary Differential Equations (Neural ODEs) to model and learn one-parameter transformations.  
- Regularizations to ensure the discovery of meaningful, non-trivial symmetries.

The method is demonstrated on tasks involving image data and partial differential equations (PDEs), showcasing its ability to extract symmetries that enhance downstream tasks such as data augmentation and neural operator learning.

For further details, refer to the full paper:  [Learning Infinitesimal Generators of Continuous Symmetries from Data](https://arxiv.org/abs/2410.21853).
---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kogyeonghoon/learning-symmetry-from-scratch.git
   cd learning-symmetry-from-scratch

2. **Create and activate the virtual environment**:
    '''bash
    conda create --name learning_symmetry python=3.10 -y
    conda activate learning_symmetry

3. **Install dependencies**:
    '''bash
    pip install -r requirements.txt

## Citation

If you find our work and/or our code useful, please cite us via:

@article{ko2024learning,
  title     = {Learning Infinitesimal Generators of Continuous Symmetries from Data},
  author    = {Gyeonghoon Ko and Hyunsu Kim and Juho Lee},
  journal   = {NeurIPS},
  year      = {2024},
  url       = {https://github.com/kogyeonghoon/learning-symmetry-from-scratch.git}
}