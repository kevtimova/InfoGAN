from __future__ import print_function
from __future__ import absolute_import
from infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli

import tensorflow as tf
import gflags
import os
import sys
import infogan.misc.datasets as datasets
from basicprop.noise import set_uniform_noise
from basicprop.datasets import FG_PIXEL, BG_PIXEL
from infogan.models.regularized_gan import RegularizedGAN
from infogan.algos.infogan_trainer import InfoGANTrainer
from infogan.misc.utils import mkdir_p
import dateutil
import dateutil.tz
import datetime


FLAGS = gflags.FLAGS


if __name__ == "__main__":

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    # Debug settings.
    gflags.DEFINE_boolean("preview", False, "Set to True to print images.")

    # Experiment naming.
    gflags.DEFINE_string("experiment_name", "", "")
    gflags.DEFINE_string("dataset", "MNIST", "[MNIST] MNIST|LINE|RECTS")
    gflags.DEFINE_string("noise", "NONE", "[NONE] FG|BG|BOTH")

    # Optimization settings.
    gflags.DEFINE_integer("batch_size", 128, "SGD minibatch size.")
    gflags.DEFINE_integer("updates_per_epoch", 1, "")
    gflags.DEFINE_integer("max_epoch", 50, "")
    gflags.DEFINE_float("info_reg_coeff", 1.0, "Hyperparameter for Mutual Information.")
    gflags.DEFINE_float("generator_learning_rate", 1e-3, "")
    gflags.DEFINE_float("discriminator_learning_rate", 2e-4, "")

    FLAGS(sys.argv)

    if not FLAGS.experiment_name:
        FLAGS.experiment_name = "{}_{}".format(FLAGS.dataset, timestamp)


    root_log_dir = "logs/mnist"
    root_checkpoint_dir = "ckt/mnist"

    log_dir = os.path.join(root_log_dir, FLAGS.experiment_name)
    checkpoint_dir = os.path.join(root_checkpoint_dir, FLAGS.experiment_name)

    mkdir_p(log_dir)
    mkdir_p(checkpoint_dir)

    if FLAGS.dataset == 'MNIST':
        dataset = datasets.MnistDataset()
    elif FLAGS.dataset == 'LINE':
        dataset = datasets.BasicPropLineDataset()
    elif FLAGS.dataset == 'RECTS':
        dataset = datasets.BasicPropRectsDataset()
    else:
        raise Exception("Please specify a valid dataset.")

    def fg_noise(x):
        x = set_uniform_noise(x, 150, 240, FG_PIXEL)
        return x

    def bg_noise(x):
        x = set_uniform_noise(x, 0, 60, BG_PIXEL)
        return x

    def both_noise(x):
        x = fg_noise(x)
        x = bg_noise(x)
        return x

    if FLAGS.noise == 'FG':
        noise_fn = fg_noise
    elif FLAGS.noise == 'BG':
        noise_fn = bg_noise
    elif FLAGS.noise == 'BOTH':
        noise_fn = both_noise
    elif FLAGS.noise == 'NONE':
        noise_fn = lambda x: x
    else:
        raise Exception("Not implemented.")

    if FLAGS.preview:
        import scipy.misc
        x = dataset.train.next_batch(FLAGS.batch_size)[0]
        x = noise_fn(x)
        x = x.reshape(FLAGS.batch_size * 28, 28)
        scipy.misc.imsave('preview.png', x)
        sys.exit()

    latent_spec = [
        (Uniform(62), False),
        (Categorical(10), True),
        (Uniform(1, fix_std=True), True),
        (Uniform(1, fix_std=True), True),
    ]

    model = RegularizedGAN(
        output_dist=MeanBernoulli(dataset.image_dim),
        latent_spec=latent_spec,
        batch_size=FLAGS.batch_size,
        image_shape=dataset.image_shape,
        network_type="mnist",
    )

    algo = InfoGANTrainer(
        model=model,
        dataset=dataset,
        noise_fn=noise_fn,
        batch_size=FLAGS.batch_size,
        exp_name=FLAGS.experiment_name,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        max_epoch=FLAGS.max_epoch,
        updates_per_epoch=FLAGS.updates_per_epoch,
        info_reg_coeff=FLAGS.info_reg_coeff,
        generator_learning_rate=FLAGS.generator_learning_rate,
        discriminator_learning_rate=FLAGS.discriminator_learning_rate,
    )

    algo.train()
