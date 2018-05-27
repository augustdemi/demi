from keras.engine.topology import Layer
import tensorflow as tf

class logREG(Layer):
    def __init__(self, num_outputs, **kwargs):
        '''
        '''

        self.num_outputs = num_outputs
        super(logREG, self).__init__(**kwargs)

    def build(self, input_shape): #(10,2000)
        '''
        '''

        self.w = self.add_weight(
                shape = [input_shape[1], self.num_outputs], #(2000,1)
                initializer = 'he_normal',
                trainable = True
                )

        self.b = self.add_weight(
                shape = [self.num_outputs], #(1)
                initializer='zero',
                trainable = True
                )


        super(logREG, self).build(input_shape)

    def call(self, x, mask=None):
        '''
        '''
        # add dimension for samples
        # var_w = self.w[None, :, :] #(n,2000,1), n=1
        # var_b = self.b[None, :] #(n,1)
        # var_x = x[:,:, None] #(10,2000,n)

        score =tf.matmul(x, self.w) + self.b
        # score = tf.reduce_sum(var_w*var_x,1) + var_b
        print(score)
        score2 = tf.sigmoid(score)
        print(score2)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> logREG")
        return score2

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_outputs)

