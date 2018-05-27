from tensorflow.contrib.slim import losses 
import tensorflow as tf

def mse(y_true, y_pred):
    '''
    '''
    # check if all values are -1
    skip = tf.not_equal(y_true, -1)
    skip = tf.reduce_min(tf.to_int32(skip),1)
    skip = tf.to_float(skip)

    cost = tf.reduce_mean(tf.square(y_true-y_pred),1)

    return cost * skip

if __name__=='__main__':

    import numpy as np
    np.random.seed(1)
    y_true = np.random.randint(0,6,[100,5])
    # y_true = -np.ones_like(y_true)
    y_pred = np.random.randint(0,6,[100,5])
    y_true[:3,:]=-1
    print(y_true[:4])

    y_true_pl = tf.placeholder(tf.float32, y_true.shape)
    y_pred_pl = tf.placeholder(tf.float32, y_true.shape)

    loss = mse(y_true_pl, y_pred_pl)

    with tf.Session() as sess:

        feed_dict ={
                y_pred_pl:y_pred,
                y_true_pl:y_true,
                }

        out = sess.run(loss, feed_dict=feed_dict)
        print(out)
        # print(out.shape)
        # print(out.mean())
