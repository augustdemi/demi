import EmoEstimator as EE
import keras.backend as KB
from keras.layers import Dense, Lambda, Input, Reshape
import keras as K
import numpy as np
import os

class VAE:
    def __init__(self, img_shape, batch_size, num_au):
        latent_dim1 = 2048
        latent_dim2 = 500
        latent_dim3 = 300
        num_of_intensity = 2

        inp_0 = Input(shape=img_shape)
        emb, shape = EE.networks.encoder(inp_0, norm=1)

        from numpy import prod
        from keras.layers import Dropout
        n_feat = prod(shape)

        emb = Dropout(0.5)(emb)

        latent_feat = Dense(latent_dim1, activation='relu', name='latent_feat')(emb)  # into 2048
        intermediate = Dense(latent_dim2, activation='relu', name='intermediate')(latent_feat)  # into 500
        z_mean = Dense(latent_dim3, name='z_mean')(intermediate)  # into latent_dim = 300은. output space의 dim이 될것.
        z_log_sigma = Dense(latent_dim3)(intermediate)

        # print('==============================')
        # print('emb', emb.shape)
        # print('latent_feat', latent_feat.shape)
        # print('intermediate', intermediate.shape)
        # print('z_mean', z_mean.shape)
        # print('z_log_sigma', z_log_sigma.shape)
        def sampling(args):  ########### input param의 평균과 분산에 noise(target_mean, sd 기준)가 섞인 샘플링 값을줌
            z_mean, z_log_sigma = args
            epsilon = []
            for m, s in zip(np.zeros(latent_dim3), np.ones(latent_dim3)):
                epsilon.append(KB.random_normal(shape=[batch_size, 1], mean=m, std=s))
            epsilon = KB.concatenate(epsilon, 1)
            return z_mean + KB.exp(z_log_sigma) * epsilon

        z = Lambda(sampling, output_shape=(latent_dim3,))([z_mean, z_log_sigma])  # 발굴한 feature space에다 노이즈까지 섞어서 샘플링한 z
        out_1 = EE.layers.softmaxPDF(num_au, num_of_intensity)(Reshape((latent_dim3, 1))(z_mean))
        D1 = Dense(latent_dim2, activation='relu')  # into 500
        D2 = Dense(latent_dim1, activation='relu')  # into 2048x
        D3 = Dense(n_feat, activation='sigmoid')  # into 2400
        h_decoded1 = D1(z)  # latent space에서 샘플링한 z를 인풋으로하여 아웃풋도 latent space인 fullyconnected layer
        h_decoded2 = D2(h_decoded1)
        x_decoded_mean = D3(h_decoded2)

        out_0 = EE.networks.decoder(x_decoded_mean, shape, norm=1)

        model_train = K.models.Model([inp_0], [out_0, out_1, out_0])
        model_deep_feature = K.models.Model([inp_0], [latent_feat])

        print("#### from build_vae #####")
        model_train.summary()

        ################# Above: to load the model.
        self.model_train = model_train
        self.model_deep_feature = model_deep_feature