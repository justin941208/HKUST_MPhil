import os
from time import perf_counter as timer
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils import *
from models import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Training
LR = 5e-3
BATCH_SIZE = 100
N_EPOCHS = 20
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Config(object):
    def __init__(self):
        # RP parameters
        self.USE_G = False
        self.G_SIZE = 1500
        self.rp_type = None

        # Training parameters
        self.LR = LR
        self.LR_DECAY = 0.5
        self.DECAY_AFTER = 2400
        self.OPTIMIZER = 'mom'
        self.WD = False
        self.BN = False

# Load the data
print('Loading data ... ')
t0 = timer()
x_train, y_train, x_test, y_test = cifar_data()
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
data = (x_train, y_train, x_test, y_test)
print('\tCompleted. Time Elapsed: {:.4f}s.'.format(timer()-t0))

# Build and run the model
print('Building graph ... ')
t0 = timer()
config = Config()
if config.USE_G is True:
    print('G size: {}'.format(config.G_SIZE))
g = tf.Graph()
run_meta = tf.RunMetadata()
with g.as_default():
    model = FCModel(3072, 4096, 10, config, g, run_meta).build()
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
