import numpy as np
from infogan.misc.utils import mkdir_p
from tensorflow.examples.tutorials import mnist
import os
import random
from basicprop.datasets import Line, Rects
from basicprop.noise import set_uniform_noise


class Dataset(object):
    def __init__(self, images, labels=None):
        self._images = images.reshape(images.shape[0], -1)
        self._labels = labels
        self._epochs_completed = -1
        self._num_examples = images.shape[0]
        # shuffle on first run
        self._index_in_epoch = self._num_examples

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            if self._labels is not None:
                self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        if self._labels is None:
            return self._images[start:end], None
        else:
            return self._images[start:end], self._labels[start:end]


class BasicPropAngleDataset(object):
    def __init__(self):
        data_directory = "BASICPROP-angle"
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        dataset = mnist.input_data.read_data_sets(data_directory)
        self.train = dataset.train
        # make sure that each type of digits have exactly 10 samples
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state()
        np.random.seed(0)
        for cat in range(10):
            ids = np.where(self.train.labels == cat)[0]
            np.random.shuffle(ids)
            sup_images.extend(self.train.images[ids[:10]])
            sup_labels.extend(self.train.labels[ids[:10]])
        np.random.set_state(rnd_state)
        self.supervised_train = Dataset(
            np.asarray(sup_images),
            np.asarray(sup_labels),
        )
        self.test = dataset.test
        self.validation = dataset.validation
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class BasicPropAngleNoiseDataset(object):
    def __init__(self):
        data_directory = "BASICPROP-angle-noise"
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        dataset = mnist.input_data.read_data_sets(data_directory)
        self.train = dataset.train
        # make sure that each type of digits have exactly 10 samples
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state()
        np.random.seed(0)
        for cat in range(10):
            ids = np.where(self.train.labels == cat)[0]
            np.random.shuffle(ids)
            sup_images.extend(self.train.images[ids[:10]])
            sup_labels.extend(self.train.labels[ids[:10]])
        np.random.set_state(rnd_state)
        self.supervised_train = Dataset(
            np.asarray(sup_images),
            np.asarray(sup_labels),
        )
        self.test = dataset.test
        self.validation = dataset.validation
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class BasicPropAngleNoiseBGDataset(object):
    def __init__(self):
        data_directory = "BASICPROP-angle-noise-bg"
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        dataset = mnist.input_data.read_data_sets(data_directory)
        self.train = dataset.train
        # make sure that each type of digits have exactly 10 samples
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state()
        np.random.seed(0)
        for cat in range(10):
            ids = np.where(self.train.labels == cat)[0]
            np.random.shuffle(ids)
            sup_images.extend(self.train.images[ids[:10]])
            sup_labels.extend(self.train.labels[ids[:10]])
        np.random.set_state(rnd_state)
        self.supervised_train = Dataset(
            np.asarray(sup_images),
            np.asarray(sup_labels),
        )
        self.test = dataset.test
        self.validation = dataset.validation
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class BasicPropAngleNoiseSingleDataset(object):
    def __init__(self):
        data_directory = "BASICPROP-angle-single-noise-bg"
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        dataset = mnist.input_data.read_data_sets(data_directory)
        self.train = dataset.train
        # make sure that each type of digits have exactly 10 samples
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state()
        np.random.seed(0)
        for cat in range(10):
            ids = np.where(self.train.labels == cat)[0]
            np.random.shuffle(ids)
            sup_images.extend(self.train.images[ids[:10]])
            sup_labels.extend(self.train.labels[ids[:10]])
        np.random.set_state(rnd_state)
        self.supervised_train = Dataset(
            np.asarray(sup_images),
            np.asarray(sup_labels),
        )
        self.test = dataset.test
        self.validation = dataset.validation
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


class DummyBatchIterator(object):
    def __init__(self):
        super(DummyBatchIterator, self).__init__()

    def next_batch(self, batch_size):
        data = np.zeros((batch_size, 784)).astype(np.float32)
        labels = np.zeros((batch_size,)).astype(np.uint8)
        return (data, labels)

class DummyDataset(object):
    def __init__(self):
        super(DummyDataset, self).__init__()

        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)
        self.train = DummyBatchIterator()

    def inverse_transform(self, data):
        return data

class BatchIterator(object):
    def __init__(self):
        raise Exception("Not implemented.")

    def get_image(self, y):
        return self.dataset.get_image(y)

    def next_batch(self, batch_size):
        """ First randomly select labels, then generate images
            based on labels and concatenate to create batch.
        """
        labels = np.random.randint(0, self.num_labels, batch_size).astype(np.uint8)
        data = []
        for y in labels:
            x = self.get_image(y)
            data.append(x.reshape(-1))
        data = np.concatenate([np.expand_dims(x, axis=0) for x in data], axis=0).astype(np.float32)
        return (data, labels)

class BasicPropLineBatchIterator(BatchIterator):
    def __init__(self):
        self.dataset = Line()
        self.num_labels = 10

class BasicPropRectsBatchIterator(BatchIterator):
    def __init__(self):
        self.dataset = Rects()
        self.num_labels = 100

class BasicPropLineDataset(object):
    def __init__(self):
        self.train = BasicPropLineBatchIterator()
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def inverse_transform(self, data):
        return data

class BasicPropRectsDataset(object):
    def __init__(self):
        self.train = BasicPropRectsBatchIterator()
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def inverse_transform(self, data):
        return data

class MnistDataset(object):
    def __init__(self):
        data_directory = "MNIST"
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        dataset = mnist.input_data.read_data_sets(data_directory)
        self.train = dataset.train
        # make sure that each type of digits have exactly 10 samples
        sup_images = []
        sup_labels = []
        rnd_state = np.random.get_state()
        np.random.seed(0)
        for cat in range(10):
            ids = np.where(self.train.labels == cat)[0]
            np.random.shuffle(ids)
            sup_images.extend(self.train.images[ids[:10]])
            sup_labels.extend(self.train.labels[ids[:10]])
        np.random.set_state(rnd_state)
        self.supervised_train = Dataset(
            np.asarray(sup_images),
            np.asarray(sup_labels),
        )
        self.test = dataset.test
        self.validation = dataset.validation
        self.image_dim = 28 * 28
        self.image_shape = (28, 28, 1)

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


def try_data():
    mnist = MnistDataset()
    basicprop = BasicPropDataset()

def load_data():
    dev_file = 'MNIST/t10k-images-idx3-ubyte'
    dev_data = convert_from_file(dev_file)

    dev_file = 'MNIST/t10k-labels-idx1-ubyte'
    dev_labels = convert_from_file(dev_file)

    train_file = 'MNIST/train-images-idx3-ubyte'
    train_data = convert_from_file(train_file)

    train_file = 'MNIST/train-labels-idx1-ubyte'
    train_labels = convert_from_file(train_file)


if __name__ == '__main__':

    import scipy.misc

    os.makedirs('img')

    bpl = BasicPropLineDataset()
    bpr = BasicPropRectsDataset()

    def preview(name, dataset, labels):
        images = [dataset.train.get_image(y) for y in labels]
        for x, y in zip(images, labels):
            scipy.misc.imsave('img/{}_{:03}.png'.format(name, y), x)

    # Generate all possible images.
    preview("line", bpl, range(10))
    preview("rects", bpr, range(100))
