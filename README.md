# InfoGAN

Code for reproducing key results in the project [Understanding Mutual Information and its use in InfoGAN](http://mrdrozdov.com/papers/infogan.pdf).

Based on the repo for the paper [InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets](https://arxiv.org/abs/1606.03657) by Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel.

## Datasets

This code use the "basicprop" synthetic dataset. Please see [http://github.com/mrdrozdov/basicprop](http://github.com/mrdrozdov/basicprop) for more details.

## Dependencies

This project has been tested with tensorflow 0.12.0, which is different from the original repo.

To install the rest of the dependencies, simply do:

```
virtualenv .venv -p python2.7
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Experiment

```bash
PYTHONPATH='.' python launchers/run_mnist_exp.py --dataset RECTS --noise both
```

You can launch TensorBoard to view the generated images:

```bash
tensorboard --logdir logs/mnist
```

## License

MIT
