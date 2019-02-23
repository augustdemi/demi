from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D, BatchNormalization, RepeatVector
from keras.models import Sequential
from ..layers import linREG, softmaxPDF, ordinalPDF

def LINEAR_REG(nb_outputs, dropout=0.5, norm=1, fcl=[512, 512]):
    def _network(net):
        for l in fcl:
            net=Dense(l)(net)
            if norm:net=BatchNormalization(axis=-1)(net)
            if dropout:net=Dropout(dropout)(net)

        net = linREG(nb_outputs)(net)
        return net
    return _network

def SOFTMAX_PDF(nb_outputs, nb_classes, dropout=0.5, norm=1, fcl=[512, 512]):
    def _network(net):
        for l in fcl:
            net=Dense(l)(net)
            if norm:net=BatchNormalization(axis=-1)(net)
            if dropout:net=Dropout(dropout)(net)

        net = softmaxPDF(nb_outputs, nb_classes)(net)
        return net
    return _network

def ORDINAL_PDF(nb_outputs, nb_classes, dropout=0.5, norm=1, fcl=[512, 512]):
    def _network(net):
        for l in fcl:
            net=Dense(l)(net)
            if norm:net=BatchNormalization(axis=-1)(net)
            if dropout:net=Dropout(dropout)(net)

        net = ordinalPDF(nb_outputs, nb_classes)(net)
        return net
    return _network
