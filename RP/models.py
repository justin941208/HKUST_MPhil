from time import perf_counter as timer
import numpy as np
import tensorflow as tf
from layers import *
from utils import get_batches

class BaseModel(object):
    # Time monitoring
    def __init__(self):
        self.step_times = []
        self.epoch_times = []
        self.test_losses = []
        self.test_accuracies = []
        self.built = False

    def add_loss(self):
        loss_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.labelsholder)
        cross_entropy_mean = tf.reduce_mean(loss_per_example,
            name='cross_entropy_loss')
        tf.add_to_collection('losses', cross_entropy_mean)
        self.loss = tf.add_n(tf.get_collection('losses'), 'total_loss')

    def add_acc_op(self):
        self.acc_op = tf.reduce_mean(tf.cast(tf.equal(
            self.labelsholder, self.predictions), tf.float32), name='accuracy')

    def add_train_op(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(self.c.LR,
                                             self.global_step,
                                             self.c.DECAY_AFTER,
                                             self.c.LR_DECAY,
                                             staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if self.c.OPTIMIZER == 'mom':
                op = tf.train.MomentumOptimizer(self.lr, 0.9)
            else:
                op = tf.train.AdamOptimizer(self.lr)
            self.train_op = op.minimize(self.loss, global_step=self.global_step)

    def build(self):
        self.add_loss()
        self.add_acc_op()
        self.add_train_op()
        self.built = True
        return self

    def train_epoch(self, sess, batches, epoch):
        if not self.built:
            raise Exception('Model has not been built yet!!!')
        t_epoch_start = timer()
        while len(batches) > 0:
            batch = batches.pop()
            feed = {self.inputsholder: batch[0],
                    self.labelsholder: batch[1],
                    self.training: 1}
            t_step_start = timer()
            sess.run(self.train_op, feed_dict=feed)
            step_time = (timer() - t_step_start) * 1000 # in milliseconds
            self.step_times.append(step_time)
            gs = sess.run(self.global_step)
            avg_step_time = np.mean(self.step_times)
            print('TRAINING: epoch = {}, global step = {}, '.format(epoch+1, gs) +
                  'step time = {:5.2f}ms'.format(step_time) +
                  ', average step time = {:5.2f}ms.'.format(avg_step_time), end='\r')
        epoch_time = timer() - t_epoch_start
        self.epoch_times.append(epoch_time)
        avg_epoch_time = np.mean(self.epoch_times)
        a_train, l_train = sess.run([self.acc_op, self.loss], feed_dict=feed)
        print('\n\tEpoch {} finished: '.format(epoch+1) +
              'batch accuracy = {:.2%}, '.format(a_train) +
              'batch loss = {:.4f}.'.format(l_train))

    def evaluate(self, sess, batches):
        if not self.built:
            raise Exception('Model has not been built yet!!!')
        l_test = []
        a_test = []
        while len(batches) > 0:
            batch = batches.pop()
            test_feed = {self.inputsholder: batch[0],
                         self.labelsholder: batch[1],
                         self.training: 0}
            l, a = sess.run([self.loss, self.acc_op], feed_dict=test_feed)
            l_test.append(l)
            a_test.append(a)
        test_loss = np.mean(l_test)
        test_accuracy = np.mean(a_test)
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_accuracy)
        print('test loss = {:.4}, test accuracy = {:.2%}'.format(
            test_loss, test_accuracy))

    def run(self, data, n_epochs, batch_size):
        x_train, y_train, x_test, y_test = data
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for j in range(n_epochs):
                batches = get_batches(x_train, y_train, batch_size=batch_size)
                self.train_epoch(sess, batches, epoch=j)
                print('\tEVALUATION: ', end='')
                test_batches = get_batches(x_test, y_test,
                                           batch_size=batch_size,
                                           shuffle=False)
                self.evaluate(sess, test_batches)

class CNNModel(BaseModel):
    def __init__(self, im_size, in_channels, n_classes, config, g, run_meta):
        BaseModel.__init__(self)
        self.c = config
        inputs_shape = [None, in_channels, im_size, im_size]
        self.inputsholder = tf.placeholder(tf.float32, shape=inputs_shape)
        self.labelsholder = tf.placeholder(tf.int64, shape=[None])
        self.training = tf.placeholder(tf.bool)

        in_layer = self.inputsholder

        if self.c.RP is True:
            self.conv1 = rp_conv_layer_v2(in_layer, self.c.NUM_RP, 128, 5, 'conv1')
        else:
            self.conv1 = conv_layer(in_layer, 64, 5, 'conv1')
        if self.c.BN is True:
            self.conv1 = tf.layers.batch_normalization(self.conv1,
                                                       axis=1,
                                                       training=self.training)
        self.pool1 = pool_layer(tf.nn.relu(self.conv1), 3, 2)

        if self.c.RP is True:
            self.conv2 = rp_conv_layer_v2(self.pool1, self.c.NUM_RP, 192, 5, 'conv2')
        else:
            self.conv2 = conv_layer(self.pool1, 96, 5, 'conv2')
        if self.c.BN is True:
            self.conv2 = tf.layers.batch_normalization(self.conv2,
                                                       axis=1,
                                                       training=self.training)
        self.pool2 = pool_layer(tf.nn.relu(self.conv2), 3, 2)


        if self.c.RP is True:
            self.conv3 = rp_conv_layer_v2(self.pool2, self.c.NUM_RP, 256, 5, 'conv3')
        else:
            self.conv3 = conv_layer(self.pool2, 128, 5, 'conv3')
        if self.c.BN is True:
            self.conv3 = tf.layers.batch_normalization(self.conv3,
                                                       axis=1,
                                                       training=self.training)
        self.pool3 = pool_layer(tf.nn.relu(self.conv3), 3, 2)


        dim = np.prod(self.pool3.shape[1:]).value
        reshape = tf.reshape(self.pool3, [-1, dim])
        print('\tConvolutional layer output dimension: {}'.format(dim))

        self.dense = dense_layer(reshape, 512, 'dense', self.c.WD)
        if self.c.BN is True:
            self.dense = tf.layers.batch_normalization(self.dense,
                                                       training=self.training)
        self.dense = tf.nn.relu(self.dense)

        self.logits = dense_layer(self.dense, n_classes, 'logits')
        self.predictions = tf.argmax(self.logits, axis=1)

class FCModel(BaseModel):
    def __init__(self, in_size, h_size, n_classes, config, g, run_meta):
        BaseModel.__init__(self)
        self.c = config
        self.inputsholder = tf.placeholder(tf.float32, shape=[None, in_size])
        self.labelsholder = tf.placeholder(tf.int64, shape=[None])
        self.training = tf.placeholder(tf.bool)

        self.h1 = dense_layer(self.inputsholder,
                              h_size,
                              'h1',
                              wd=self.c.WD,
                              use_g=self.c.USE_G,
                              g_size=self.c.G_SIZE,
                              rp_type=self.c.rp_type)
        if self.c.BN is True:
            self.h1 = tf.layers.batch_normalization(self.h1,
                                                    training=self.training)
        self.h1 = tf.nn.relu(self.h1)

        self.h2 = dense_layer(self.h1,
                              h_size,
                              'h2',
                              wd=self.c.WD,
                              use_g=self.c.USE_G,
                              g_size=self.c.G_SIZE,
                              rp_type=self.c.rp_type)
        if self.c.BN is True:
            self.h2 = tf.layers.batch_normalization(self.h2,
                                                    training=self.training)
        self.h2 = tf.nn.relu(self.h2)

        self.logits = dense_layer(self.h2,
                                  n_classes,
                                  'logits')
        self.predictions = tf.argmax(self.logits, axis=1)

        '''
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)

        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        params = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)

        print('Total FLOPS: {}.'.format(flops.total_float_ops) +
              '\nTotal params: {}.'.format(params.total_parameters))
        '''
