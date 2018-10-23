import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback 
from ..utils.evaluate import print_summary

from skimage.io import imsave

class summary_vac(Callback):
    '''
    '''
    def __init__(self, 
            gen,
            predictor,
            log_dir = '/tmp/multi_output/',
            nb_batches = 10,
            batch_size = 16,
            ):

        self.gen = gen
        self.predictor = predictor
        self.log_dir = log_dir
        self.batch_size = batch_size


        # generate data from N batchs
        X, Y = [], []
        for i in range(nb_batches):
            x, y = next(gen)
            X.append(x)
            Y.append(y)

        #        print("1111")
        #        print(Y)

        # transpose and concatonate datasets
        Y = list(map(list, zip(*Y)))
        #        print("22222")
        #        print(Y)
        Y = [np.vstack(i) for i in Y]

        X = list(map(list, zip(*X)))
        X = [np.vstack(i) for i in X]

        self.X = X
        self.Y = Y


    def on_epoch_end(self, epoch, logs={}):
        REC, Z, Y_hat = self.predictor(self.X, batch_size=self.batch_size)
        print()
        print('mean---->',Z.mean())
        print('var----->',Z.var())

        # ==============================================
        path = self.log_dir + "/latent.pkl"

        #       print("================ sum vac ===============")
        #       print(self.Y[1])
        #       print(Y_hat)
        import pickle
        z_out = open(path, 'wb')
        pickle.dump({'z': Z, 'y': self.Y[1], 'y_pred': Y_hat}, z_out, protocol=2)

        # ==============================================



        OUT = []
        for img, rec in zip(self.X[0], REC):

            img = img-img.min()
            img = img/img.max()

            rec= rec-rec.min()
            rec= rec/rec.max()

            out = np.concatenate((img, rec), 1)[:,:,0]
            OUT.append(out)
        OUT = np.concatenate(OUT,0)
        imsave(self.log_dir+'/'+str(epoch).zfill(6)+'.png',OUT)
