# MNIST data preparation

Run `python prepare_datasets.py mnist`. The loader downloads the archive linked by the [Keras MNIST loader](https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py) and checks its published SHA-256:

```text
731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1
```

It concatenates 60,000 original training images followed by 10,000 original test images, preserves raw uint8 pixel values in [0, 255], flattens 28×28 images in C order, and writes `data/prepared/mnist.npz` with `X` of shape `(70000, 784)` and `y` of shape `(70000,)`. The adjacent JSON records source, checksum, dimensions, class counts and conversion steps.

For this SEC experiment, new non-stratified splits use 5% seen and 95% unseen samples. The original MNIST train/test separation is not used for evaluation. Matching the paper's size does not prove identical author preprocessing or row order.

The Keras documentation attributes MNIST to Yann LeCun and Corinna Cortes and lists CC BY-SA 3.0; see the linked source and the [MNIST homepage](http://yann.lecun.com/exdb/mnist/). Dataset files are not redistributed here. Local downloads and prepared arrays are ignored by Git.

The existing preparation module and its other dataset functions are retained unchanged to preserve source provenance. Only MNIST is presented in this repository. The module imports Pillow for its image helpers, so `requirements-data.txt` includes that dependency even though the MNIST conversion itself uses NumPy.
