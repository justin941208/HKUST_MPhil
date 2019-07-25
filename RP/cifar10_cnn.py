import os
from time import perf_counter as timer
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils import cifar_data
from models import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

LR = 1e-2
BATCH_SIZE = 100
N_EPOCHS = 20
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Config(object):
    def __init__(self):
        # RP parameters
        self.RP = False
        self.NUM_RP = 55
        self.rp_type = None

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
x_train, y_train, x_test, y_test = cifar_data()
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
x_train = x_train.reshape(-1, 3, 32, 32)#.transpose(0, 2, 3, 1)
x_test = x_test.reshape(-1, 3, 32, 32)#.transpose(0, 2, 3, 1)

data = (x_train, y_train, x_test, y_test)
print('\tCompleted. Time Elapsed: {:.4f}s.'.format(timer()-t0))

# Build and run the model
print('Building graph ... ')
t0 = timer()
config = Config()
if config.RP == True:
    print('RP size: {}'.format(config.NUM_RP))
g = tf.Graph()
run_meta = tf.RunMetadata()
with g.as_default():
    model = CNNModel(32, 3, 10, config, g, run_meta).build()
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
