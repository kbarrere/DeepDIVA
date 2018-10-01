# DeepDIVA: A Highly-Functional Python Framework for Reproducible Experiments

[Original Repository](https://github.com/DIVA-DIA/DeepDIVA)

[This repository main page](https://github.com/kbarrere/DeepDIVA) 

## Handwritten Text Recognition Branch

The branch [htr](https://github.com/kbarrere/DeepDIVA/tree/htr) aims at developping Handwritten Text Recognition models and training within DeepDIVA.

## Installation

Clone the Github repository

``` shell
git clone https://github.com/kbarrere/DeepDIVA.git
```

Switch to the branch [htr](https://github.com/kbarrere/DeepDIVA/tree/htr) with PyTorch 0.4 compatibility :

``` shell
git chekout htr
```

It is using a fixed commit from the original repository (kind of a Beta build for PyTorch 0.4)

Then run the script to setup the environment :

``` shell
bash setup_environment.sh
```

Reload your environment variables from `.bashrc` with: `source ~/.bashrc`

## Verifying Everything Works

To verify the correctness of the procecdure you can run a small experiment. Activate the DeepDIVA python environment:

``` shell
source activate deepdiva
```

Download the MNIST dataset:

``` shell
python util/data/get_a_dataset.py mnist --output-folder toy_dataset
```

Train a simple Convolutional Neural Network on the MNIST dataset using the command:

``` shell
python template/RunMe.py --output-folder log --dataset-folder toy_dataset/MNIST --lr 0.1 --ignoregit --disable-dataset-integrity  --no-cuda
```

