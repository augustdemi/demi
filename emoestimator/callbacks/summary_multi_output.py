import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback 
from ..utils.evaluate import print_summary

from skimage.io import imsave

class summary_multi_output(Callback):
    '''
    '''
    def __init__(self, 
            gen_list,
            predictor,
            title = None,
            log_dir = '/tmp/multi_output/',
            one_hot = False,
            nb_batches = 25,
            batch_size = 32,
            ):

        self.gen_list = gen_list
        self.predictor = predictor
        self.title = title 
        self.one_hot = one_hot
        self.log_dir = log_dir
        self.batch_size = batch_size
        if title!=None:
            self.title=title
        else:
            self.title=['dset_'+str(i) for i in range(len(gen_list))]

        self.dat = []
        for gen in gen_list:

            # generate data from N batchs
            X, Y = [], []
            for i in range(nb_batches):
                x, y = next(gen)
                X.append(x)
                Y.append(y)


            # transpose and concatonate datasets
            Y = list(map(list, zip(*Y)))
            Y = [np.vstack(i) for i in Y]
            if self.one_hot:
                for i in range(len(Y)):
                    Y[i]=Y[i].argmax(2)

            X = list(map(list, zip(*X)))
            X = [np.vstack(i) for i in X]

            self.dat.append([X,Y])

        self.nb_outputs = len(Y)
        self.nb_inputs = len(X)

        print(self.title)
        print(X)
        print(Y)

        X, Y = self.dat[0]
        out = print_summary(y[0], y[0], verbose=0)
        self.index = out['index']

    def set_model(self, model):
        self.sess = K.get_session()
        self.model  = model
        self.dict_summary = {}

        self.dict_summary={}
        for dset in self.title:
            for metric in self.index:
                para_name = dset+'/'+metric
                tmp = tf.placeholder(tf.float32) 
                self.dict_summary[para_name] = tmp
                tf.summary.scalar(para_name, tmp) 
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir)

    def on_epoch_end(self, epoch, logs={}):
        avr_values = {}
        for [X, Y], dset in zip(self.dat, self.title):
            print()
            print(dset)

            Y_hat = self.predictor(X,batch_size=self.batch_size)

            if type(Y_hat) is not list:Y_hat = [Y_hat]
            if self.one_hot:
                for i in range(len(Y_hat)):
                    Y_hat[i]=Y_hat[i].argmax(2)
            for i, [y_hat, y_lab] in enumerate(zip(Y_hat, Y)):
                print(i)
                print(" >>>>>>>> y hat ")
                print(y_hat)
                print(" >>>>>>>> y lab ")
                print(y_lab)
                out = print_summary(
                    y_hat, y_lab,
                    verbose=1,
                    log_dir=self.log_dir + '/' + dset + '_' + str(i) + '_' + str(epoch).zfill(4) + '.txt'
                        )

                for metric, values in zip(out['index'], out['data']):
                    para_name = dset+'/'+metric
                    try:
                        avr_values[para_name]+=values.mean()
                    except KeyError:
                        avr_values[para_name]=values.mean()

        tensor_list = []
        feed_dict = {}
        for key in sorted(avr_values):
            avr_values[key]=avr_values[key]/len(self.dat)

            tensor_list.append( self.dict_summary[key] )
            feed_dict[ self.dict_summary[key] ] = avr_values[key]
        
        res = self.sess.run([self.merged]+tensor_list, feed_dict=feed_dict)
        self.writer.add_summary(res[0], epoch)
        self.writer.flush()



class summary_intensity(Callback):
    '''
    '''
    def __init__(self, 
            gen,
            predictor,
            title = None,
            log_dir = '/tmp/multi_output/',
            nb_batches = 25,
            batch_size = 32,
            ):

        self.predictor = predictor
        self.title = title 
        self.log_dir = log_dir
        self.batch_size = batch_size

        # generate data from N batchs
        X, Y = [], []
        for i in range(nb_batches):
            x, y = next(gen)
            X.append(x)
            Y.append(y)

        self.Y = np.vstack(Y)
        self.X = np.vstack(X)

        self.nb_outputs = len(self.Y)
        self.nb_inputs = len(self.X)

        print(self.title)

        out = print_summary(self.Y, self.Y, verbose=0)
        self.index = out['index']

    def set_model(self, model):
        self.sess = K.get_session()
        self.model  = model
        self.dict_summary = {}

        self.dict_summary={}
        for metric in self.index:
            para_name = self.title+'/'+metric
            tmp = tf.placeholder(tf.float32) 
            self.dict_summary[para_name] = tmp
            tf.summary.scalar(para_name, tmp) 

        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir)

    def on_epoch_end(self, epoch, logs={}):
        print("====== y hat / self.y =====")
        Y_hat = self.predictor.predict(self.X, batch_size=self.batch_size)
        print(Y_hat)
        print(self.Y)
        out = print_summary(Y_hat, self.Y, verbose=1)

        tensor_list = []
        feed_dict = {}
        for metric, values in zip(out['index'], out['data']):
            para_name = self.title+'/'+metric
            tensor_list.append( self.dict_summary[para_name] )
            feed_dict[ self.dict_summary[para_name] ] = values.mean()

        res = self.sess.run([self.merged]+tensor_list, feed_dict=feed_dict)
        self.writer.add_summary(res[0], epoch)
        self.writer.flush()

class summary_variable(Callback):
    '''
    '''
    def __init__(self, 
            gen,
            predictor,
            title = None,
            log_dir = '/tmp/multi_output/',
            nb_batches = 25,
            batch_size = 32,
            ):

        self.predictor = predictor
        self.title = title 
        self.log_dir = log_dir
        self.batch_size = batch_size

        # generate data from N batchs
        X = []
        for i in range(nb_batches):
            x = next(gen)
            X.append(x)

        self.X = np.vstack(X)
        self.nb_inputs = len(self.X)


    def on_epoch_end(self, epoch, logs={}):
        print()
        Z = self.predictor.predict(self.X, batch_size=self.batch_size)
        print('max',Z.max())
        print('min',Z.min())
        print('mean',Z.mean())
        print('std',Z.std())
