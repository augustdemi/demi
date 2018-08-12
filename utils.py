""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Image helper
def get_images(path, label_int, seed, nb_samples=None, validate=False):
    subject = int(path[-1])

    # random seed는 subject에 따라서만 다르도록. 즉, 한 subject내에서는 k가 증가해도 계속 동일한 seed인것.
    def sampler(path, label):
        img_path_list = os.listdir(os.path.join(path, label))
        if validate:
            random.seed(subject + seed + 10)
        else:
            random.seed(subject + seed)
        if len(img_path_list) < nb_samples:
            print('nb_samples: ', nb_samples)
            print('len img_path_list: ', len(img_path_list))
            print('img_path_list: ', img_path_list)
            random_imgs = img_path_list  # 일단 가진 on img다 때려넣고
            print('>>>> random_imgs: ', random_imgs)
            img_path_list = os.listdir(os.path.join(path, 'off'))  # off img dir에 가서
            print('after off, len img_path_list: ', len(img_path_list))
            print('nb_samples - already chosed on imgs: ', nb_samples - len(random_imgs))
            random_imgs = random_imgs.extend(
                random.sample(img_path_list, nb_samples - len(random_imgs)))  # 나머지는 off로 채워넣음
            print('>>>> random_imgs: ', random_imgs)

        else:
            random_imgs = random.sample(img_path_list, nb_samples)
        return random_imgs
    labels = ['off', 'on'] # off = 0, on =1
    #각 task별로 k*2개 씩의 label 과 img담게됨. path = till subject.
    images = [(i, os.path.join(path,label, image)) \
        for i, label in zip(label_int, labels) \
              for image in sampler(path, label)]
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


def xent_sig(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size
