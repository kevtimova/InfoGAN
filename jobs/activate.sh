#!/bin/bash

module load cuda/7.5.18
# module load cudnn/7.0v4.0
module load numpy/intel/1.10.1
module load pandas/intel/0.17.1
module load scikit-learn/intel/0.18
module load tensorflow/python2.7/20161029

MODEL_NAME="infogan"

PREV_DIR=`pwd`

cd
cd InfoGAN
. .venv-hpc/bin/activate
cd $PREV_DIR

export PYTHONPATH=$PYTHONPATH:"."

