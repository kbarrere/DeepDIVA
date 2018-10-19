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

Go inside the cloned repository

``` shell
cd DeepDIVA
```

Switch to the branch [htr](https://github.com/kbarrere/DeepDIVA/tree/htr) with PyTorch 0.4 compatibility :

``` shell
git checkout htr
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

You can try to do a training on a GPU by removing the `--no-cuda` flag.

## Common Installation problems

If when trying to run a script you end up with an error that looks like `ImportError: No module names '<module name>'` , a solution is to export the current directory to the list of python scripts :

``` shell
export PYTHONPATH="/path/to/DeepDiva:$PYTHONPATH"
```

## Running The Handwritten Text Recognition Task

If everything work, you can start to work with this command (still in development) :

``` shell
python template/RunMe.py --runner-class handwritten_text_recognition --output-folder log --piff-json <path/to/file.json> --lr 0.1 --ignoregit --disable-dataset-integrity --model-name crnn --batch-size 256 --epochs 20000
```
