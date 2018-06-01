import EmoEstimator as EE
import EmoData as ED
import keras.backend as KB
from keras.layers import Dense, Lambda, Input, Reshape
import keras as K
import numpy as np
import h5py
import argparse
from datetime import datetime
import tensorflow as tf


class VAE:

    def __init__(self, img_shape, label_shape):

        self.img_shape = img_shape,
        self.label_shape =label_shape,

        w_1 = 1 / 50
        batch_size=2
        latent_dim = 2000
        target_std_vec = np.ones(latent_dim)
        target_mean_vec = np.zeros(latent_dim)





        print("img_shape", self.img_shape[0])
        print("label_shape", self.label_shape[0])
        img_shape = self.img_shape[0]
        label_shape = self.label_shape[0]
        inp_0       = Input(shape=img_shape)
        emb, shape  = EE.networks.encoder(inp_0, norm=1)

        from numpy import prod
        from keras.layers import Dropout
        print("shape before flatten: ", shape)
        n_feat = prod(shape)
        print("shape after flatten: ", emb.get_shape(), n_feat)


        emb = Dropout(0.5)(emb)
        print("emb: ", emb)
        z_mean      = Dense(latent_dim)(emb) # latent_dim는 output space의 dim이 될것. activation함수는 none임 out_1(라벨값 y)을 위한 layer쌓는중  classifier?????????????
        z_log_sigma = Dense(latent_dim)(emb) #

        def sampling(args): ########### input param의 평균과 분산에 noise(target_mean, sd 기준)가 섞인 샘플링 값을줌
            z_mean, z_log_sigma = args
            epsilon = []
            for m, s in zip(target_mean_vec, target_std_vec):
                epsilon.append(KB.random_normal(shape=[batch_size, 1], mean=m, std=s))
            epsilon = KB.concatenate(epsilon, 1)
            return z_mean + KB.exp(z_log_sigma) * epsilon

        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma]) # 발굴한 feature space에다 노이즈까지 섞어서 샘플링한 z

        print("Z: ", z)
        print(z.get_shape())
        resized_z = Reshape((2000, 1))(z_mean)
        out_1 = EE.layers.softmaxPDF(label_shape[0], label_shape[1])(resized_z)

        D1 = Dense(latent_dim, activation='relu')
        D2 = Dense(n_feat, activation='sigmoid')  # n_feat  = conv 결과 shape들의 곱이 ouputspace의 dim
        h_decoded = D1(z) # latent space에서 샘플링한 z를 인풋으로하여 아웃풋도 latent space인 fullyconnected layer

        x_decoded_mean = D2(h_decoded)

        out_0 = EE.networks.decoder(x_decoded_mean, shape, norm=1) # 위에서만든 layer로 디코더 실행. 근데 사실상 이 디코더에 오기까지 오리지날 트레인 x를 인코드하는거부터 시작됨. vae.


        def vae_loss(img, rec):
            print("img", img)
            print("rec", rec)
            print(">>>>>>>>>> vae loss")
            kl_loss = - 0.5 * KB.mean(1 + z_log_sigma - KB.square(z_mean) - KB.exp(z_log_sigma), axis=-1)
            return w_1*kl_loss

        def rec_loss(img, rec):
            print("img", img)
            print("rec", rec)
            print(">>>>>>>>>> rec loss")
            mse = EE.losses.mse(img, rec)
            return mse

        def pred_loss(y_true, y_pred):
            print("y_true", y_true)
            print("y_pred", y_pred)
            print(">>>>>>>>> pred loss")
            ce = EE.losses.categorical_crossentropy(y_true, y_pred)
            return (1 - w_1) * ce

        loss = [rec_loss, pred_loss, vae_loss]

        model_train = K.models.Model([inp_0], [out_0, out_1, out_0]) #inp_0: train data, out_0 : reconstruted img, out_1: predicted label. (vae)에서 쌓은 레이어로 모델만듦
        model_z_int = K.models.Model([inp_0], [z_mean, out_1])

        self.model_train = model_train
        self.model_z_int = model_z_int
        self.z = z




    def computeLatentVal(self, x):
        self.model_train.load_weights("./model78.h5")
        z, pred = self.model_z_int.predict(x, batch_size=len(x))
        return self.model_train.get_weights()[-2:], z
