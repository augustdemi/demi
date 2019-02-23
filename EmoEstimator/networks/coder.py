from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D, BatchNormalization, UpSampling2D
from keras.layers import Reshape


def encoder(net, norm=0):
    '''
    '''
    net = Convolution2D(64, 5, 5, activation='relu', border_mode='same')(net)
    if norm:net=BatchNormalization(axis=-1)(net)
    net = MaxPooling2D(pool_size=[2,2])(net)

    net = Convolution2D(48, 5, 5, activation='relu', border_mode='same')(net)
    if norm:net=BatchNormalization(axis=-1)(net)
    net = MaxPooling2D(pool_size=[2,2])(net)

    net = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(net)
    if norm:net=BatchNormalization(axis=-1)(net)
    net = MaxPooling2D(pool_size=[2,2])(net)

    net = Convolution2D(16, 5, 5, activation='relu', border_mode='same')(net)
    if norm:net=BatchNormalization(axis=-1)(net)
    net = MaxPooling2D(pool_size=[2,2])(net)
    print(net)

    out_shape = [i.value for i in net.get_shape()[1:]]

    net = Flatten()(net)

    return net, out_shape

def decoder(net, shape, norm=0):
    '''
    '''
    net = Reshape(shape)(net)
    net = UpSampling2D((2, 2))(net)
    net = Convolution2D(16, 5, 5, activation='relu', border_mode='same')(net)
    if norm:net=BatchNormalization(axis=-1)(net)

    net = UpSampling2D((2, 2))(net)
    net = Convolution2D(32, 5, 5, activation='relu', border_mode='same')(net)
    if norm:net=BatchNormalization(axis=-1)(net)

    net = UpSampling2D((2, 2))(net)
    net = Convolution2D(48, 5, 5, activation='relu', border_mode='same')(net)
    if norm:net=BatchNormalization(axis=-1)(net)

    net = UpSampling2D((2, 2))(net)
    net = Convolution2D(64, 5, 5, activation='relu', border_mode='same')(net)
    if norm:net=BatchNormalization(axis=-1)(net)

    net = Convolution2D(1, 5, 5, activation='relu', border_mode='same')(net)

    return net
