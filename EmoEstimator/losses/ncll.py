from tensorflow.contrib.slim import losses 
import tensorflow as tf
import keras as K

def categorical_crossentropy(y_true, y_pred):

    '''
    '''
    # check if all values are -1
    skip = tf.not_equal(y_true, -1)
    skip = tf.reduce_min(tf.to_int32(skip), 1)
    skip = tf.reduce_min(tf.to_int32(skip), 1)[:, None]
    skip = tf.to_float(skip)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return loss * skip
