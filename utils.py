""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Image helper
def get_images(path, label_int, nb_samples=None, shuffle=False):
    if shuffle:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x[:nb_samples]
    labels = ['off', 'on'] # off = 0, on =1
    #각 task별로 k*2개 씩의 label 과 img담게됨. path = till subject.
    images = [(i, os.path.join(path,label, image)) \
        for i, label in zip(label_int, labels)\
        for image in sampler(os.listdir(os.path.join(path, label)))]

    if shuffle:
        random.shuffle(images)
    return images

## Network helpers
def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID', residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    if FLAGS.max_pool:
        conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    else:
        conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
    normed = normalize(conv_output, activation, reuse, scope)
    if FLAGS.max_pool:
        normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
    return normed

def normalize(inp, activation, reuse, scope):
    if FLAGS.norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size

