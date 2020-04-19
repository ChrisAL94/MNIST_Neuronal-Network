# MNIST_Neuronal-Network
*Code for Neuronal Network for handwritten digit recognition*

## MNIST dataset

The MNIST database is available at http://yann.lecun.com/exdb/mnist/

The MNIST database is a dataset of handwritten digits. It has 60,000 training
samples, and 10,000 test samples. Each image is represented by 28x28 pixels, each
containing a value 0 - 255 with its grayscale value.

![](https://github.com/datapythonista/mnist/raw/master/img/samples.png)

It is a subset of a larger set available from NIST.
The digits have been size-normalized and centered in a fixed-size image.

It is a good database for people who want to try learning techniques and pattern recognition
methods on real-world data while spending minimal efforts on preprocessing and formatting.

There are four files available, which contain separately train and test, and images and labels.

Thanks to Yann LeCun, Corinna Cortes, Christopher J.C. Burges.

## Usage

mnist makes it easier to download and parse MNIST files.

To automatically download the train files, and display the first image in the
dataset, you can simply use:

```python
import mnist
import scipy.misc

images = mnist.train_images()
scipy.misc.toimage(scipy.misc.imresize(images[0,:,:] * -1 + 256, 10.))
```

## Software and Libraries

This project uses the following software and Python libraries:

* [Python](https://www.python.org/downloads/release/python-364/)
* [NumPy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/)
* [matplotlib](https://matplotlib.org/)
