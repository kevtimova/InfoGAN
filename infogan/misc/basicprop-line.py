from infogan.misc.utils import mkdir_p
import os
import numpy as np
import random


def create_data(shuffle=True):
    """ TODO:
        - [x] Shuffle the data
        - [x] Randomize the pixels within some range
    """

    from idx2numpy import convert_to_file

    width = 4

    train_data = []
    train_labels = []
    train_size = 10000
    for i in range(10):
        x = np.zeros((train_size,28,28), dtype=np.uint8)
        offset = i * 2
        line = np.random.randint(150, 240, (train_size,28,width))
        x[:,:,(offset+3):(offset+3+width)] = line
        train_data.append(x)
        y = np.zeros((train_size,), dtype=np.uint8)
        y[:] = i
        train_labels.append(y)

    train_data = np.concatenate(train_data, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)

    if shuffle:
        perm = range(train_data.shape[0])
        random.shuffle(perm)
        train_data = train_data[perm]
        train_labels = train_labels[perm]

    eval_data = []
    eval_labels = []
    eval_size = 1000
    for i in range(10):
        x = np.zeros((eval_size,28,28), dtype=np.uint8)
        offset = i * 2
        line = np.random.randint(150, 240, (eval_size,28,width))
        x[:,:,(offset+3):(offset+3+width)] = line
        eval_data.append(x)
        y = np.zeros((eval_size,), dtype=np.uint8)
        y[:] = i
        eval_labels.append(y)

    eval_data = np.concatenate(eval_data, axis=0)
    eval_labels = np.concatenate(eval_labels, axis=0)

    if shuffle:
        perm = range(eval_data.shape[0])
        random.shuffle(perm)
        eval_data = eval_data[perm]
        eval_labels = eval_labels[perm]

    mkdir_p('BASICPROP-line')
    convert_to_file('BASICPROP-line/t10k-images-idx3-ubyte', eval_data)
    convert_to_file('BASICPROP-line/t10k-labels-idx1-ubyte', eval_labels)
    convert_to_file('BASICPROP-line/train-images-idx3-ubyte', train_data)
    convert_to_file('BASICPROP-line/train-labels-idx1-ubyte', train_labels)


if __name__ == '__main__':

    create_data()
