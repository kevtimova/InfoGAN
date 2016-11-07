from infogan.misc.utils import mkdir_p
import os
import numpy as np
import random


cBLACK = 0
cWHITE = 255


def create_data(shuffle=True, use_noise=False, bg_noise=False, output='BASICPROP-angle', single=False):
    """ TODO:
        - [x] Shuffle the data
        - [x] Randomize the pixels within some range
    """

    from idx2numpy import convert_to_file

    width = 10
    x1offset = 3
    x2offset = 15
    heights = np.array([3, 5, 7, 9, 11, 13, 15, 17, 19, 21])

    train_data = []
    train_labels = []
    train_size = 1000
    for i in range(10):
        for j in range(10):
            if bg_noise:
                x = np.random.randint(0, 60, (train_size,28,28), dtype=np.uint8)
            else:
                x = np.zeros((train_size,28,28), dtype=np.uint8)
            
            if use_noise:
                line1 = np.random.randint(150, 240, (train_size, heights[i], width))
            else:
                line1 = np.ones((train_size, heights[i], width))
            x[:,3:3+heights[i],x1offset:x1offset+width] = line1
            
            if not single:
                if use_noise:
                    line2 = np.random.randint(150, 240, (train_size, heights[j], width))
                else:
                    line2 = np.ones((train_size, heights[j], width))
                x[:,3:3+heights[j],x2offset:x2offset+width] = line2
            
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
    eval_size = 100
    for i in range(10):
        for j in range(10):
            if bg_noise:
                x = np.random.randint(0, 60, (eval_size,28,28), dtype=np.uint8)
            else:
                x = np.zeros((eval_size,28,28), dtype=np.uint8)

            if use_noise:
                line1 = np.random.randint(150, 240, (eval_size, heights[i], width))
            else:
                line1 = np.ones((eval_size, heights[i], width))
            x[:,3:3+heights[i],x1offset:x1offset+width] = line1

            if not single:
                if use_noise:
                    line2 = np.random.randint(150, 240, (eval_size, heights[j], width))
                else:
                    line2 = np.ones((eval_size, heights[j], width))
                x[:,3:3+heights[j],x2offset:x2offset+width] = line2

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

    if single:
        output = '{}-single'.format(output)

    if use_noise:
        output = '{}-noise'.format(output)

    if bg_noise:
        output = '{}-bg'.format(output)

    import ipdb; ipdb.set_trace()

    mkdir_p(output)
    convert_to_file('{}/t10k-images-idx3-ubyte'.format(output), eval_data)
    convert_to_file('{}/t10k-labels-idx1-ubyte'.format(output), eval_labels)
    convert_to_file('{}/train-images-idx3-ubyte'.format(output), train_data)
    convert_to_file('{}/train-labels-idx1-ubyte'.format(output), train_labels)


if __name__ == '__main__':
    import gflags
    import sys
    
    FLAGS = gflags.FLAGS

    gflags.DEFINE_bool("single", False, "")
    gflags.DEFINE_bool("shuffle", True, "")
    gflags.DEFINE_bool("use_noise", False, "")
    gflags.DEFINE_bool("bg_noise", False, "")
    gflags.DEFINE_string("output", "BASICPROP-angle", "")

    FLAGS(sys.argv)

    create_data(FLAGS.shuffle, FLAGS.use_noise, FLAGS.bg_noise, FLAGS.output, FLAGS.single)
