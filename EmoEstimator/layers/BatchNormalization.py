from keras.engine.topology import Layer
import tensorflow as tf
import tensorflow.contrib.slim as slim

class BatchNormalization(Layer):
    def __init__(self, **kwargs):
        '''
        '''
        super(BatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        '''
        super(BatchNormalization, self).build(input_shape)

    def call(self, x, mask=None):
        '''
        '''
        # input_shape = [i.value for i in x.get_shape()[1:]]
        # nb_features = 1
        # for i in input_shape:nb_features*=i
        # x = tf.reshape(x, [-1, nb_features])
        
        # u = tf.reduce_mean(x, axis=1, keep_dims=True)
        # v =  tf.reduce_mean((x-u)**2, axis=1, keep_dims=True)

        # x = (x-u)/v
        x = slim.batch_norm(x)
        # x = tf.reshape(x, [-1]+input_shape)


        return x 

    def get_output_shape_for(self, input_shape):
        return input_shape

class MY_BN(Layer):
    def __init__(self, **kwargs):
        '''
        '''
        super(MY_BN, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        '''
        super(MY_BN, self).build(input_shape)

    def call(self, x, mask=None):
        '''
        '''
        input_shape = [i.value for i in x.get_shape()[1:]]
        nb_features = 1
        for i in input_shape:nb_features*=i
        x = tf.reshape(x, [-1, nb_features])
        
        u = tf.reduce_mean(x, axis=1, keep_dims=True)
        v =  tf.reduce_mean((x-u)**2, axis=1, keep_dims=True)
        x = (x-u)/(v**0.5)
        x = tf.reshape(x, [-1]+input_shape)

        return x 

    def get_output_shape_for(self, input_shape):
        return input_shape

