#!/bin/bash

module load cuda-7.5
module load cudnn-v4-cuda7

PYTHONPATH='.' python launchers/run_mnist_exp.py --experiment_name line --dataset LINE
PYTHONPATH='.' python launchers/run_mnist_exp.py --experiment_name line-fg --dataset LINE --noise FG
PYTHONPATH='.' python launchers/run_mnist_exp.py --experiment_name line-bg --dataset LINE --noise BG
PYTHONPATH='.' python launchers/run_mnist_exp.py --experiment_name line-both --dataset LINE --noise BOTH

PYTHONPATH='.' python launchers/run_mnist_exp.py --experiment_name rects --dataset RECTS
PYTHONPATH='.' python launchers/run_mnist_exp.py --experiment_name rects-fg --dataset RECTS --noise FG
PYTHONPATH='.' python launchers/run_mnist_exp.py --experiment_name rects-bg --dataset RECTS --noise BG
PYTHONPATH='.' python launchers/run_mnist_exp.py --experiment_name rects-both --dataset RECTS --noise BOTH

