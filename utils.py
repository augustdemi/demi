""" Utility functions. """
import math
import os
import random
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

## Image helper
def get_images(path, label_int, seed, nb_samples=None, validate=True):
    subject = int(path[-1])
    print("============================================")
    print(">>>>>>>>>>>>>subject: ", subject)
    # random seed는 subject에 따라서만 다르도록. 즉, 한 subject내에서는 k가 증가해도 계속 동일한 seed인것.
    labels = ['off', 'on']  # off = 0, on =1

    # check count of existing samples
    num_existing_samples = []
    for label in labels:
        img_path_list = os.listdir(os.path.join(path, label))
        num_existing_samples.append(len(img_path_list))

    # make the balance
    num_samples_to_select = [nb_samples, nb_samples]
    if num_existing_samples[0] < nb_samples:
        n_off_samples = 2 * math.floor(num_existing_samples[0] / 2)
        n_on_samples = nb_samples - n_off_samples
        num_samples_to_select = [n_off_samples, n_on_samples]
    elif num_existing_samples[1] < nb_samples:
        n_on_samples = 2 * math.floor(num_existing_samples[1] / 2)
        n_off_samples = nb_samples - n_on_samples
        num_samples_to_select = [n_off_samples, n_on_samples]
    print('num_samples_to_select: ', num_samples_to_select)

    def sampler(path, label, n_samples):
        print("-------------------------------")
        print("validate: ", validate)
        print("label: ", label)

        img_path_list = os.listdir(os.path.join(path, label))
        if validate:
            random.seed(subject + seed + 10)
        else:
            random.seed(subject + seed)
        random_imgs = random.sample(img_path_list, n_samples)
        random_img_path = [os.path.join(path, label, img) for img in random_imgs]
        return random_img_path

    #각 task별로 k*2개 씩의 label 과 img담게됨. path = till subject.
    off_images = sampler(path, labels[0], num_samples_to_select[0])
    print('num of off_images: ', len(off_images))
    on_images = sampler(path, labels[1], num_samples_to_select[1])
    print('num of on_images: ', len(on_images))
    return off_images, on_images


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
