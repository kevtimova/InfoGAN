#!/bin/bash
#PBS -l nodes=1:ppn=4
#PBS -N infogan
#PBS -j oe
#PBS -M apd283@nyu.edu
#PBS -l mem=6GB
#PBS -l walltime=4:00:00

module load cuda/7.5.18
# module load cudnn/7.0v4.0
module load numpy/intel/1.10.1
module load pandas/intel/0.17.1
module load scikit-learn/intel/0.18
module load tensorflow/python2.7/20161029

MODEL_NAME="infogan"

cd
cd InfoGAN
. .venv-hpc/bin/activate

export PYTHONPATH=$PYTHONPATH:"."

export MODEL_FLAGS=" \
--experiment_name $MODEL_NAME \
"

echo "python launchers/run_mnist_exp.py $MODEL_FLAGS"

python launchers/run_mnist_exp.py $MODEL_FLAGS

