import numpy as np
import scipy as sp
import tensorflow as tf

def rand_ortho(in_size, out_size):
    a = np.random.randn(in_size, out_size)
    u, s, v = sp.linalg.svd(a)
    while np.any(np.isclose(s, 0)):
        a = np.random.randn(in_size, out_size)
    q, r = np.linalg.qr(a)
    return q

def pool_layer(in_tensor, size, stride, padding='SAME'):
    return tf.nn.max_pool(in_tensor,
                          [1, 1, size, size],
                          [1, 1, stride, stride],
                          'SAME',
                          data_format='NCHW')

def conv_layer(in_tensor, n_filter, filter_size, name, stride=1):
    in_channel = in_tensor.shape[1].value
    weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
    with tf.variable_scope(name) as scope:
        kernel = tf.get_variable('kernel',
                                 [filter_size, filter_size, in_channel, n_filter],
                                 dtype=tf.float32,
                                 initializer=weight_init)
        bias = tf.get_variable('bias',
                               [n_filter],
                               dtype=tf.float32,
                               initializer=tf.zeros_initializer())
    conv = tf.nn.conv2d(in_tensor,
                        kernel,
                        strides=[1, 1, stride, stride],
                        padding='SAME',
                        data_format='NCHW')
    return tf.nn.bias_add(conv, bias, data_format='NCHW')

def sep_conv(in_tensor, n_filter, filter_size, name, stride=1):
    in_channel = in_tensor.shape[-1].value
    weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
    with tf.variable_scope(name) as scope:
        ker_dw = tf.get_variable('ker_dw',
                                 [filter_size, filter_size, in_channel, 1],
                                 dtype=tf.float32,
                                 initializer=weight_init)
        ker_pw = tf.get_variable('ker_pw',
                                 [1, 1, in_channel, n_filter],
                                 dtype=tf.float32,
                                 initializer=weight_init)
        bias = tf.get_variable('bias',
                               [n_filter],
                               dtype=tf.float32,
                               initializer=tf.zeros_initializer())
    conv = tf.nn.separable_conv2d(in_tensor,
                                  ker_dw,
                                  ker_pw,
                                  strides=[1, 1, 1, 1],
                                  padding='SAME')
    return tf.nn.bias_add(conv, bias)

def dense_layer(in_tensor,
                out_size,
                name,
                wd=False,
                use_g=False,
                g_size=None,
                rp_type=None):
    with tf.variable_scope(name) as scope:
        if use_g is True:
            in_size = in_tensor.shape[-1].value
            if rp_type == 'o':
                g = rand_ortho(in_size, g_size) * np.sqrt(in_size / g_size)
            elif rp_type == 'u':
                g = np.random.randn(in_size, g_size)
                g /= np.linalg.norm(g, axis=0)
            else:
                g = np.random.randn(in_size, g_size) / np.sqrt(g_size)
            g = tf.constant(g, dtype=tf.float32)
            x = tf.matmul(in_tensor, g, name='gauss')
        else:
            x = in_tensor
        w_size = x.shape[-1].value
        weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
        w = tf.get_variable('w',
                            [w_size, out_size],
                            dtype=tf.float32,
                            initializer=weight_init)
        b = tf.get_variable('b',
                            [out_size],
                            dtype=tf.float32,
                            initializer=tf.zeros_initializer())
        if wd is True:
            weight_decay = tf.multiply(tf.nn.l2_loss(w), 0.004, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return tf.add(tf.matmul(x, w), b)

def rp_conv_layer_v1(in_tensor,
                     n_rp_filter,
                     n_filter,
                     filter_size,
                     name,
                     stride=1):
    print('\n\tWarning: using a slower version of rp_von_layer.\n')
    _, size, _, in_channel = in_tensor.shape
    a = np.random.randn(filter_size * filter_size * in_channel.value, n_rp_filter)
    const = (size.value ** 2) * n_rp_filter
    a /= np.sqrt(const)
    rp_filters = tf.constant(a, dtype=tf.float32)
    weight_init = tf.truncated_normal_initializer(stddev=np.sqrt(2/n_rp_filter),
                                                  dtype=tf.float32)
    filter_shape = [filter_size, filter_size, in_channel.value, n_filter]
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('w',
                            [n_rp_filter, n_filter],
                            dtype=tf.float32,
                            initializer=weight_init)
        kernel = tf.reshape(tf.matmul(rp_filters, w), filter_shape)
        bias = tf.get_variable('bias',
                               [n_filter],
                               dtype=tf.float32,
                               initializer=tf.zeros_initializer())
        conv = tf.nn.conv2d(in_tensor,
                            kernel,
                            strides=[1, stride, stride, 1],
                            padding='SAME')
        return tf.nn.relu(tf.nn.bias_add(conv, bias))

def rp_conv_layer_v2(in_tensor,
                     n_rp_filter,
                     n_filter,
                     filter_size,
                     name,
                     stride=1):
    _, in_channel, size, _ = in_tensor.shape
    a = np.random.randn(filter_size, filter_size, in_channel.value, n_rp_filter)
    rp_filters = tf.constant(a, dtype=tf.float32)
    conv_rp = tf.nn.conv2d(in_tensor,
                           rp_filters,
                           strides=[1, 1, stride, stride],
                           padding='SAME',
                           data_format='NCHW')
    with tf.variable_scope(name) as scope:
        weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
        w = tf.get_variable('w',
                            [1, 1, n_rp_filter, n_filter],
                            dtype=tf.float32,
                            initializer=weight_init)
        bias = tf.get_variable('bias',
                               [n_filter],
                               dtype=tf.float32,
                               initializer=tf.zeros_initializer())
    conv = tf.nn.conv2d(conv_rp,
                        w,
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                        data_format='NCHW')
    return tf.nn.bias_add(conv, bias, data_format='NCHW')

def rp_conv_layer_v2a(in_tensor,
                      n_rp_filter,
                      n_filter,
                      filter_size,
                      name,
                      stride=1):
    _, size, _, in_channel = in_tensor.shape
    a = np.random.randn(filter_size, filter_size, in_channel.value, n_rp_filter)
    rp_filters = tf.constant(a, dtype=tf.float32)
    conv_rp = tf.nn.conv2d(in_tensor,
                           rp_filters,
                           strides=[1, stride, stride, 1],
                           padding='SAME')
    conv_rp = tf.reshape(conv_rp, [-1, n_rp_filter])
    final_shape = [-1, size, size, n_filter]
    with tf.variable_scope(name) as scope:
        weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
        w = tf.get_variable('w',
                            [n_rp_filter, n_filter],
                            dtype=tf.float32,
                            initializer=weight_init)
        bias = tf.get_variable('bias',
                               [n_filter],
                               dtype=tf.float32,
                               initializer=tf.zeros_initializer())
    conv = tf.reshape(tf.matmul(conv_rp, w), final_shape)
    return tf.nn.bias_add(conv, bias)

def rp_conv_layer_v3(in_tensor,
                     n_rp_filter,
                     n_filter,
                     filter_size,
                     name,
                     stride=1):
    _, in_channel, size, _ = in_tensor.shape
    a = np.random.randn(filter_size, filter_size, n_rp_filter)
    basis = tf.constant(a, dtype=tf.float32)
    filter_shape = [filter_size, filter_size, in_channel.value, n_filter]
    with tf.variable_scope(name) as scope:
        weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
        w = tf.get_variable('w',
                            [n_rp_filter, in_channel.value, n_filter],
                            dtype=tf.float32,
                            initializer=weight_init)
        filters = tf.einsum('hwm,mcn->hwcn', basis, w)
        bias = tf.get_variable('bias',
                               [n_filter],
                               dtype=tf.float32,
                               initializer=tf.zeros_initializer())
    conv = tf.nn.conv2d(in_tensor,
                        filters,
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                        data_format='NCHW')
    return tf.nn.bias_add(conv, bias, data_format='NCHW')

def rp_conv_layer_v3a(in_tensor,
                      n_rp_filter,
                      n_filter,
                      filter_size,
                      name,
                      stride=1):
    _, in_channel, size, _  = in_tensor.shape
    a = np.random.randn(filter_size, filter_size, 1, n_rp_filter)
    a = a.repeat(in_channel.value, axis=2)
    a = tf.constant(a, dtype=tf.float32)
    with tf.variable_scope(name) as scope:
        weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
        w = tf.get_variable('w',
                            [1, 1, n_rp_filter * in_channel.value, n_filter],
                            dtype=tf.float32,
                            initializer=weight_init)
        bias = tf.get_variable('bias',
                               [n_filter],
                               dtype=tf.float32,
                               initializer=tf.zeros_initializer())
    conv = tf.nn.separable_conv2d(in_tensor,
                                  a,
                                  w,
                                  strides=[1, 1, 1, 1],
                                  padding='SAME',
                                  data_format='NCHW')
    return tf.nn.bias_add(conv, bias, data_format='NCHW')

def rp_conv_layer_v3b(in_tensor,
                      n_rp_filter,
                      n_filter,
                      filter_size,
                      name,
                      stride=1):
    n, in_channel, size, _ = in_tensor.shape.as_list()
    in_tensor = tf.expand_dims(in_tensor, axis=1)
    print(in_tensor.shape.as_list())
    a = np.random.randn(1, filter_size, filter_size, 1, n_rp_filter)
    a = tf.constant(a, dtype=tf.float32)
    print(a.shape.as_list())
    mid = tf.nn.conv3d(in_tensor,
                       a,
                       strides=[1, 1, 1, stride, stride],
                       padding='SAME',
                       data_format='NCDHW')
    print(mid.shape.as_list())
    n_slices = np.prod(mid.shape.as_list()[1:3])
    mid = tf.reshape(mid, [-1, n_slices] + mid.shape.as_list()[-2:])
    print(mid.shape.as_list())
    with tf.variable_scope(name) as scope:
        weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
        w = tf.get_variable('w',
                            [1, 1, n_rp_filter * in_channel, n_filter],
                            dtype=tf.float32,
                            initializer=weight_init)
        bias = tf.get_variable('bias',
                               [n_filter],
                               dtype=tf.float32,
                               initializer=tf.zeros_initializer())
    conv = tf.nn.conv2d(mid,
                        w,
                        strides=[1, 1, 1, 1],
                        padding='SAME',
                        data_format='NCHW')
    return tf.nn.bias_add(conv, bias, data_format='NCHW')

def rp_conv_layer_v4(in_tensor,
                     n_rp_filter,
                     n_filter,
                     filter_size,
                     name,
                     stride=1):
    _, size, _, in_channel = in_tensor.shape
    a = np.random.randn(1, filter_size, filter_size, in_channel.value * n_rp_filter)
    a = tf.constant(a, dtype=tf.float32)
    rp_filters_shape = [filter_size, filter_size, in_channel.value, n_rp_filter]
    with tf.variable_scope(name) as scope:
        weight_init = tf.contrib.layers.xavier_initializer(uniform=False)
        w1 = tf.get_variable('w1',
                             [1, 1, in_channel.value * n_rp_filter, 1],
                             dtype=tf.float32,
                             initializer=weight_init)
        w2 = tf.get_variable('w2',
                             [1, 1, n_rp_filter, n_filter],
                             dtype=tf.float32,
                             initializer=weight_init)
        bias = tf.get_variable('bias',
                               [n_filter],
                               dtype=tf.float32,
                               initializer=tf.zeros_initializer())
    rp_filters = tf.nn.depthwise_conv2d(a,
                                        w1,
                                        strides=[1, 1, 1, 1],
                                        padding='SAME')
    rp_filters = tf.reshape(rp_filters, rp_filters_shape)
    conv_rp = tf.nn.conv2d(in_tensor,
                           rp_filters,
                           strides=[1, stride, stride, 1],
                           padding='SAME')
    conv = tf.nn.conv2d(conv_rp,
                        w2,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    return tf.nn.bias_add(conv, bias)

def rp_layer(in_tensor, rp_size, stride, padding, const=None):
    _, in_size, _, in_channels = in_tensor.shape
    if const is None:
        const = (np.ceil((in_size.value - rp_size) / stride) + 1) ** 2 * in_channels.value
    #const = np.prod([rp_size, rp_size, in_channels.value])
    print('\tNormalization constant: {}'.format(const))
    g = np.random.randn(rp_size, rp_size, in_channels, in_channels) / np.sqrt(const)
    g = tf.constant(g, dtype=tf.float32)
    return tf.nn.conv2d(in_tensor,
                        g,
                        strides=[1, stride, stride, 1],
                        padding=padding)
