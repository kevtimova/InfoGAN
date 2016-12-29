from __future__ import print_function
from __future__ import absolute_import
from infogan.misc.distributions import Uniform, Categorical, Gaussian, MeanBernoulli

import tensorflow as tf
import gflags
import os
import sys
import infogan.misc.datasets as datasets
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

    # Experiment naming.
    gflags.DEFINE_string("experiment_name", "", "")
    gflags.DEFINE_string("dataset", "MNIST", "[MNIST] MNIST|BASICPROP|BPAngle|BPAngleNoise")

    # Optimization settings.
    gflags.DEFINE_integer("batch_size", 128, "SGD minibatch size.")
    gflags.DEFINE_integer("updates_per_epoch", 100, "")
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
    elif FLAGS.dataset == 'DUMMY':
        dataset = datasets.DummyDataset()
    elif FLAGS.dataset == 'LINE':
        dataset = datasets.BasicPropLineDataset()
    elif FLAGS.dataset == 'RECTS':
        dataset = datasets.BasicPropRectsDataset()
    elif FLAGS.dataset == 'BPAngle':
        dataset = datasets.BasicPropAngleDataset()
    elif FLAGS.dataset == 'BPAngleNoise':
        dataset = datasets.BasicPropAngleNoiseDataset()
    elif FLAGS.dataset == 'BPAngleNoiseBG':
        dataset = datasets.BasicPropAngleNoiseBGDataset()
    elif FLAGS.dataset == 'BPAngleNoiseSingle':
        dataset = datasets.BasicPropAngleNoiseSingleDataset()
    elif FLAGS.dataset == 'BPAngleNoiseSingle':
        dataset = datasets.BasicPropAngleNoiseSingleDataset()
    else:
        raise Exception("Please specify a valid dataset.")

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
