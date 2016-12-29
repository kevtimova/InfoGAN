import unittest

import numpy as np
import infogan.misc.datasets as datasets

def check_dataset(dataset):
    dummy = 1.0
    batch_size = 10
    assert dataset.image_dim == 784
    assert dataset.inverse_transform(dummy) == dummy
    assert dataset.image_shape == (28, 28, 1)
    
    batch = dataset.train.next_batch(batch_size)
    assert len(batch) == 2
    assert batch[0].shape == (batch_size, 784)
    assert batch[0].dtype == np.float32
    assert batch[1].shape == (batch_size,)
    assert batch[1].dtype == np.uint8

class DataTestCase(unittest.TestCase):

    def test_mnist(self):
        dataset = datasets.MnistDataset()
        check_dataset(dataset)

    def test_dummy(self):
        dataset = datasets.DummyDataset()
        check_dataset(dataset)

    def test_basicprop(self):
        dataset = datasets.DummyDataset()
        check_dataset(dataset)


if __name__ == '__main__':
    unittest.main()
