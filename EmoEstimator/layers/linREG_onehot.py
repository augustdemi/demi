from keras.engine.topology import Layer
import tensorflow as tf

class linREG_onehot(Layer):
    def __init__(self, num_outputs, **kwargs):
        '''
        '''

        self.num_outputs = num_outputs
        super(linREG_onehot, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        '''

        self.w = self.add_weight( 
                shape = [12,2000,6],
                initializer = 'he_normal',
                trainable = True
                )

        self.b = self.add_weight( 
                shape = [12,6],
                initializer='zero',
                trainable = True
                )


        super(linREG_onehot, self).build(input_shape)

    def call(self, x, mask=None):
        '''
        '''
        # add dimension for samples
        var_w = self.w[:, None, :, :]
        var_b = self.b[:, None, :]
        var_x = x[:,:, None]

        s = var_w*var_x
        t = tf.reduce_sum(var_w*var_x,2)
        onehot_score = t + var_b
        # softmax = tf.nn.softmax(onehot_score)
        # print(softmax)
        pred_label = tf.arg_max(onehot_score,2)
        pred_label =tf.cast(pred_label, tf.float32)
        pred_label = tf.reshape(pred_label, [-1,12])
        print(">>>>>>>>>>>>>>>>> linREG_onehot")
        return pred_label

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_outputs)

