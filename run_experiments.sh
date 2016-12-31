#!/bin/bash

module load cuda-7.5
module load cudnn-v4-cuda7

PYTHONPATH='.' python launchers/run_mnist_exp.py --experiment_name line --dataset LINE
PYTHONPATH='.' python launchers/run_mnist_exp.py --experiment_name line-fg --dataset LINE --noise fg
PYTHONPATH='.' python launchers/run_mnist_exp.py --experiment_name line-bg --dataset LINE --noise bg
PYTHONPATH='.' python launchers/run_mnist_exp.py --experiment_name line-both --dataset LINE --noise both

PYTHONPATH='.' python launchers/run_mnist_exp.py --experiment_name rects --dataset RECTS 
PYTHONPATH='.' python launchers/run_mnist_exp.py --experiment_name rects-fg --dataset RECTS --noise fg
PYTHONPATH='.' python launchers/run_mnist_exp.py --experiment_name rects-bg --dataset RECTS --noise bg
PYTHONPATH='.' python launchers/run_mnist_exp.py --experiment_name rects-both --dataset RECTS --noise both

