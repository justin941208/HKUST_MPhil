import os
from time import perf_counter as timer
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#import matplotlib.pyplot as plt
from utils import *
from models import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

LR = 1e-2
BATCH_SIZE = 100
N_EPOCHS = 20
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Config(object):
    def __init__(self):
        # RP parameters
        self.RP = True
        self.NUM_RP = 15

        # Training parameters
        self.LR = LR
        self.LR_DECAY = 0.5
        self.DECAY_AFTER = 2400
        self.OPTIMIZER = 'adam'
        self.WD = False
        self.BN = True

# Load the data
print('Loading data ... ')
t0 = timer()
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
x_train = np.concatenate([mnist.train.images, mnist.validation.images], axis=0)
y_train = np.concatenate([mnist.train.labels, mnist.validation.labels], axis=0)
y_train = np.argmax(y_train, axis=1)
x_test = mnist.test.images
y_test = np.argmax(mnist.test.labels, axis=1)

x_train, x_test = mean_normalize(x_train, x_test)

x_train = np.reshape(x_train, [-1, 1, 28, 28])
x_test = np.reshape(x_test, [-1, 1, 28, 28])
data = (x_train, y_train, x_test, y_test)
print('\tCompleted. Time Elapsed: {:.4f} seconds.'.format(timer()-t0))

# Build and run the model
print('Building graph ... ')
t0 = timer()
config = Config()
g = tf.Graph()
run_meta = tf.RunMetadata()
with g.as_default():
    model = CNNModel(28, 1, 10, config, g, run_meta).build()
    print('\tCompleted. Time Elapsed: {:.4f}s.'.format(timer()-t0))
    print('Training ...')
    print('\tLR: {}'.format(LR))
    t0 = timer()
    model.run(data, N_EPOCHS, BATCH_SIZE)

avg_epoch = np.mean(model.epoch_times)
avg_step = np.mean(model.step_times)
last = model.test_accuracies[-1]
best = np.max(model.test_accuracies)
print('Completed. Time Elapsed: {:.4f}s.'.format(timer()-t0) +
    '\nAverage Epoch Time: {:.4f}s.'.format(avg_epoch) +
    '\nAverage Step Time: {:.2f}ms'.format(avg_step) +
    '\nBest Test Accuracy: {:.2%}'.format(best) +
    '\nFinal Test Accuracy: {:.2%}'.format(last))
