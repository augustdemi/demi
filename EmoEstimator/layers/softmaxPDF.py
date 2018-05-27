from keras.engine.topology import Layer
import tensorflow as tf

class softmaxPDF(Layer):
    def __init__(self, nb_outputs, nb_classes, **kwargs):
        '''
        '''
        self.nb_outputs = nb_outputs
        self.nb_classes = nb_classes
        super(softmaxPDF, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        '''
        self.w = self.add_weight( 
                shape = [input_shape[1], self.nb_outputs, self.nb_classes], 
                initializer = 'glorot_normal',
                trainable = True
                )

        self.b = self.add_weight( 
                shape = [self.nb_outputs, self.nb_classes], 
                initializer='zero',
                trainable = True
                )


        super(softmaxPDF, self).build(input_shape)

    def call(self, x, mask=None):
        '''
        '''
        # add dimension for samples
        var_w = self.w[ None, ::]

        # add dimension for features 
        var_b = self.b[ None, ::]

        # add dimension for output and class
        var_x = x[ :, :, None]

        # matrix multiplication with dropout
        z = tf.reduce_sum( var_w*var_x , 1) + var_b 

        # score = tf.clip_by_value(score, -20, 20)

        return tf.nn.softmax(z)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nb_outputs, self.nb_classes)
