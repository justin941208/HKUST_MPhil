import os
from time import perf_counter as timer
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#import matplotlib.pyplot as plt
from utils import *
from models import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

LR = 1e-1
BATCH_SIZE = 100
N_EPOCHS = 20
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Config(object):
    def __init__(self):
        # RP parameters
        self.USE_G = False
        self.G_SIZE = 250
        self.rp_type = None

        # Training parameters
        self.LR = LR
        self.LR_DECAY = 0.5
        self.DECAY_AFTER = 2400
        self.OPTIMIZER = 'mom'
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
data = (x_train, y_train, x_test, y_test)
print('\tCompleted. Time Elapsed: {:.4f}s.'.format(timer()-t0))

# Build and run the model
print('Building Graph ... ')
t0 = timer()
config = Config()
g = tf.Graph()
run_meta = tf.RunMetadata()
with g.as_default():
    model = FCModel(784, 1024, 10, config, g, run_meta).build()
    print('\tCompleted. Time Elapsed: {:.4f}s.'.format(timer()-t0))
    print('Training ... ')
    print('\tLR: {}\n\tOptimizer: {}'.format(LR, config.OPTIMIZER.upper()))
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


'''
x1, x2, y1, y2 = eval_epochs, test_epochs, train_accuracy, test_accuracy
if eval_every == test_every:
    fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(20,16), dpi=100)
    ax1, ax2, ax3, ax4 = axarr.flatten()
    ax4.plot(x1, y1, 'b', x1, y1, 'bs', x1, y2, 'g', x1, y2, 'g^')
    ax4.set_xticks(x1)
    ax4.set_title('Train Accuracy vs Test Accuracy', fontsize=18)
else:
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(12,16), dpi=100)
ax1.plot(x1, y1, 'b', x1, y1, 'bs')
ax1.set_title('Train Accuracy', fontsize=18)
ax1.set_xticks(x1)
ax2.plot(x2, y2, 'g', x2, y2, 'g^')
ax2.set_xticks(x2)
ax2.set_title('Test Accuracy', fontsize=18)
ax3.plot(x1, train_loss, 'r', x1, train_loss, 'rx')
ax3.set_xticks(x1)
ax3.set_title('Train Loss', fontsize=18)

plt.suptitle('MNIST Results', fontsize=22)
plt.savefig('Results_{}_{}.png'.format(n_hidden, n_G))
'''
