from keras.engine.topology import Layer
import tensorflow as tf

class ordinalPDF(Layer):
    def __init__(self, nb_outputs, nb_classes, **kwargs):
        '''
        '''
        self.nb_outputs = nb_outputs
        self.nb_classes = nb_classes

        super(ordinalPDF, self).__init__(**kwargs)

    def build(self, input_shape):
        '''
        '''

        self.a = self.add_weight( 
                shape = [input_shape[1], self.nb_outputs], 
                initializer = 'glorot_normal',
                trainable = True
                )

        self.b = self.add_weight( 
                shape = [self.nb_outputs], 
                initializer = 'zero',
                trainable = True
                )

        self.d = self.add_weight( 
                shape = [self.nb_outputs, self.nb_classes-2], 
                initializer = 'one',
                trainable = True
                )

        self.s = self.add_weight(
                shape = [self.nb_outputs], 
                initializer = 'one',
                trainable = True
                )

        super(ordinalPDF, self).build(input_shape)

    def call(self, x, mask=None):
        '''
        '''
        x = x[:, :, None]
        a = self.a[None, :, :]
        b = self.b[:,None]
        d = self.d 
        s = self.s[:,None, None]
        
        aTx = tf.reduce_sum( a*x, 1)
        aTx = tf.transpose(aTx)
        aTx = aTx[:, None]

        t = tf.cumsum( tf.concat([b, d**2.], 1), 1)
        t = t[:,:,None]

        aTx = tf.tile(aTx,[1,self.nb_classes-1,1])

        x = (t-aTx) / (s**2.)

        F = tf.nn.sigmoid(x)

        l1 = tf.ones_like(F[:,0,:][:,None])
        l0 = tf.zeros_like(F[:,0,:][:,None])

        F_0 = tf.concat([l0,F], 1)
        F_1 = tf.concat([F,l1], 1)

        pdf = F_1 - F_0 
        pdf = tf.transpose(pdf,[2,0,1])

        return pdf

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nb_outputs, self.nb_classes)


if __name__=='__main__':
    import numpy as np
    import keras as K
    X = np.random.normal(size = [10,256])
    print(X.shape)
    print(X.min())
    print(X.max())


    inp_0 = K.layers.Input( shape=X.shape[1:] )

    out = ordinalPDF(12,6)(inp_0)
    mod = K.models.Model(inp_0, out)
    pred = mod.predict(X)
    print(pred.min())
    print(pred.max())
    print(pred.mean())
