from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D, BatchNormalization, RepeatVector
from keras.models import Sequential
from keras.layers.pooling import AveragePooling2D, GlobalMaxPooling2D
# (pool_size=(2, 2), strides=None, padding='valid', data_format=None)

def CNN(norm=0, filter_size=5, layers=[32, 64, 128], fcl=[512, 256], global_pooling = False, dropout=0.5):
    '''
    '''

    def _network(net):
        f = filter_size
        for l in layers:
            net = Convolution2D(l,f,f, activation='relu', border_mode='same')(net)
            net = MaxPooling2D(pool_size=[2,2])(net)
            if norm:net = BatchNormalization(axis=-1)(net)

        if global_pooling:
            net = GlobalMaxPooling2D()(net)
        else:
            net = Flatten()(net)

        for l in fcl:
            net=Dense(l)(net)
            if norm:net=BatchNormalization(axis=-1)(net)
            if dropout:net=Dropout(dropout)(net)

        return net

    return _network
