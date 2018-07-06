from keras.callbacks import Callback
import numpy as np
from skimage.io import imsave
from keras import backend as K

class summary_rec(Callback):
    def __init__(self,
                 gen,
                 predictor,
                 log_dir='/tmp/multi_output/',
                 nb_batches=10,
                 batch_size=16,
                 ):

        self.gen = gen
        self.predictor = predictor
        self.log_dir = log_dir
        self.batch_size = batch_size

        # generate data from N batchs
        X, Y, S = [], [] ,[]
        for i in range(nb_batches):
            x, y, s = next(gen)
            X.append(x)
            S.append(s)


        X = list(map(list, zip(*X)))
        X = [np.vstack(i) for i in X]
        S = list(map(list, zip(*S)))
        S = [np.vstack(i) for i in S]

        self.S = S
        self.X = X


    def on_epoch_end(self, epoch, logs={}):
        REC, Z = self.predictor(self.X, batch_size=self.batch_size)
        print()
        print('mean---->', Z.mean())
        print('var----->', Z.var())

        # ==============================================
        path = self.log_dir + "/latent_rec.pkl"

        #       print("================ sum vac ===============")
        #       print(self.Y[1])
        #       print(Y_hat)
        import pickle
        z_out = open(path, 'wb')
        pickle.dump({'z': Z, 's':self.S}, z_out, protocol=2)

        # ==============================================

        OUT = []
        for img, rec in zip(self.X[0], REC):
            img = img - img.min()
            img = img / img.max()

            rec = rec - rec.min()
            rec = rec / rec.max()

            out = np.concatenate((img, rec), 1)[:, :, 0]
            OUT.append(out)
        OUT = np.concatenate(OUT, 0)
        imsave(self.log_dir + '/' + str(epoch).zfill(6) + '.png', OUT)
