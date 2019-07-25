import pickle
import numpy as np

def read_images(path):
    with open(path, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3 * 96 * 96)).astype(np.float64)
        return images

def read_labels(path):
    with open(path, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8).astype(np.int64) - 1
        return labels

def unpickle(f):
    with open(f, 'rb') as fo:
        out = pickle.load(fo, encoding='bytes')
    return out

def cifar_data():
    images_list = []
    labels_list = []
    for i in range(5):
        traindict = unpickle('data/data_batch_{}'.format(i+1))
        images_list.append(traindict[b'data'])
        labels_list.append(traindict[b'labels'])
    x_train = np.asarray(np.concatenate(images_list, axis=0), dtype=np.float64)
    y_train = np.asarray(np.concatenate(labels_list, axis=0), dtype=np.int64)
    testdict = unpickle('data/test_batch')
    x_test = np.asarray(testdict[b'data'], dtype=np.float64)
    y_test = np.asarray(testdict[b'labels'], dtype=np.int64)
    return x_train, y_train, x_test, y_test

def scale(f):
    l = np.min(f)
    u = np.max(f)
    return (f - l) / (u - l)

def per_sample_standardization(tensor):
    in_shape = tensor.shape
    if len(in_shape) > 2:
        tensor = np.reshape(tensor, [in_shape[0], -1])
    m = np.mean(tensor, axis=1, keepdims=True)
    std = np.std(tensor, axis=1, keepdims=True)
    adj_std = np.maximum(std, 1 / np.sqrt(tensor.shape[1]))
    if len(in_shape) > 2:
        return np.reshape((tensor - m) / adj_std, in_shape)
    else:
        return (tensor - m) / adj_std

def per_sample_normalize(x_train, x_test):
    x_train = per_sample_standardization(x_train)
    x_test = per_sample_standardization(x_test)
    return x_train, x_test

def mean_normalize(x_train, x_test):
    m = np.mean(x_train, axis=0, keepdims=True)
    return x_train - m, x_test - m

def scale_normalize(x_train, x_test, lower, upper):
    assert upper > lower
    a = upper - lower
    b = a - upper
    min = np.min(x_train, axis=0, keepdims=True)
    max = np.max(x_train, axis=0, keepdims=True)
    x_train = a * (x_train - min) / (max - min) - b
    x_test = a * (x_test - min) / (max - min) - b
    return x_train, x_test

def std_normalize(x_train, x_test):
    std = np.std(x_train, axis=0, keepdims=True)
    x_train /= std
    x_test /= std
    return x_train, x_test

def r_crop(images, size):
    n, w_im, h_im, channels = images.shape
    out = np.zeros([n, size, size, channels])
    w_ids = np.random.randint(w_im - size + 1, size=n)
    h_ids = np.random.randint(h_im - size + 1, size=n)
    for i in range(n):
        w, h = w_ids[i], h_ids[i]
        out[i] = images[i, w:w+size, h:h+size, :]
    return out

def get_batches(x, y, batch_size, shuffle=True, crop=False, crop_size=24):
    assert x.shape[0] == y.shape[0]
    if crop is True:
        x = r_crop(x, crop_size)
    if shuffle is True:
        permutation = np.random.permutation(np.arange(x.shape[0]))
        x = x[permutation]
        y = y[permutation]
    batches = []
    while True:
        if x.shape[0] <= batch_size:
            batches.append((x, y))
            break
        else:
            batches.append((x[:batch_size], y[:batch_size]))
            x = x[batch_size:]
            y = y[batch_size:]
    return batches
