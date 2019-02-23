from keras.engine.topology import Layer
import tensorflow as tf

class linREG(Layer):
    def __init__(self, num_outputs, **kwargs):
        '''
        '''

        self.num_outputs = num_outputs
        super(linREG, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        '''

        self.w = self.add_weight( 
                shape = [input_shape[1], self.num_outputs], 
                initializer = 'he_normal',
                trainable = True
                )

        self.b = self.add_weight( 
                shape = [self.num_outputs], 
                initializer='zero',
                trainable = True
                )


        super(linREG, self).build(input_shape)

    def call(self, x, mask=None):
        '''
        '''
        # add dimension for samples
        var_w = self.w[None, :, :]
        var_b = self.b[None, :]
        var_x = x[:,:, None]

        score = tf.reduce_sum(var_w*var_x,1) + var_b

        return score

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_outputs)
