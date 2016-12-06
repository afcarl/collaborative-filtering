import numpy as np
import pandas as pd
import tensorflow as tf


def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval = low, maxval = high,
                             dtype = tf.float32)


def preprocess(x):
    pre = lambda x: (float(x) - 3) / 2
    return generic_apply(pre, x)


def postprocess(x):
    post = lambda x: 2 * x + 3
    return generic_apply(post, x)


def generic_apply(func, data):
    if isinstance(data, (int, float, np.int32, np.int64, np.float32, np.float64)):
        return func(data)
    elif isinstance(data, list):
        return [func(v) for v in data]
    elif isinstance(data, np.ndarray):
        vfunc = np.vectorize(func)
        return vfunc(data)
    elif isinstance(data, pd.Series):
        return data.apply(func)
    else:
        raise TypeError('unsupported {}'.format(type(data)))