from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D, BatchNormalization
from keras.models import Model, Sequential


def mlp(norm=0, layers=[20, 100, 20]):
    '''
    '''
    def _network(net):
        for l in layers:
            net = Dense(l)(net)
            if norm:net = BatchNormalization(axis=-1)(net)
        net = Flatten()(net)
        return net

    return _network
