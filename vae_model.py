import EmoEstimator as EE
import keras.backend as KB
from keras.layers import Dense, Lambda, Input, Reshape
import keras as K
import numpy as np


class VAE:
    def __init__(self, img_shape, batch_size, num_au):

        latent_dim = 2000
        target_std_vec = np.ones(latent_dim)
        target_mean_vec = np.zeros(latent_dim)

        inp_0 = Input(shape=img_shape)
        emb, shape = EE.networks.encoder(inp_0, norm=1)

        from numpy import prod
        from keras.layers import Dropout
        n_feat = prod(shape)

        emb = Dropout(0.5)(emb)
        z_mean = Dense(latent_dim)(
            emb)  # latent_dim는 output space의 dim이 될것. activation함수는 none임 out_1(라벨값 y)을 위한 layer쌓는중  classifier?????????????
        z_log_sigma = Dense(latent_dim)(emb)  #

        def sampling(args):  ########### input param의 평균과 분산에 noise(target_mean, sd 기준)가 섞인 샘플링 값을줌
            z_mean, z_log_sigma = args
            epsilon = []
            for m, s in zip(target_mean_vec, target_std_vec):
                epsilon.append(KB.random_normal(shape=[batch_size, 1], mean=m, std=s))
            epsilon = KB.concatenate(epsilon, 1)
            return z_mean + KB.exp(z_log_sigma) * epsilon

        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])  # 발굴한 feature space에다 노이즈까지 섞어서 샘플링한 z
        resized_z = Reshape((2000, 1))(z_mean)
        out_1 = EE.layers.softmaxPDF(num_au, 2)(resized_z)

        D1 = Dense(latent_dim, activation='relu')
        D2 = Dense(n_feat, activation='sigmoid')  # n_feat  = conv 결과 shape들의 곱이 ouputspace의 dim
        h_decoded = D1(z)  # latent space에서 샘플링한 z를 인풋으로하여 아웃풋도 latent space인 fullyconnected layer

        x_decoded_mean = D2(h_decoded)

        out_0 = EE.networks.decoder(x_decoded_mean, shape,
                                    norm=1)  # 위에서만든 layer로 디코더 실행. 근데 사실상 이 디코더에 오기까지 오리지날 트레인 x를 인코드하는거부터 시작됨. vae.

        model_train = K.models.Model([inp_0], [out_0, out_1,
                                               out_0])  # inp_0: train data, out_0 : reconstruted img, out_1: predicted label. (vae)에서 쌓은 레이어로 모델만듦
        model_z_int = K.models.Model([inp_0], [z_mean, out_1])
        model_au_int = K.models.Model([inp_0], [out_1])
        self.model_train = model_train
        self.model_z_int = model_z_int
        self.model_au_int = model_au_int
        self.z = z

    def computeLatentVal(self, x, vae_model, au_idx):
        self.model_train.load_weights(vae_model)
        z, _ = self.model_z_int.predict(x, batch_size=len(x))
        loaded_weight = self.model_train.get_weights()[-2:]
        if 'all' in vae_model:
            print(">>>>>>>>>>>>>> au_idx in ", vae_model + ": ", au_idx)
            w = loaded_weight[0][:, au_idx]
            b = loaded_weight[1][au_idx]
            loaded_weight= [w.reshape(2000,1,2), b.reshape(1,2)]
        return loaded_weight, z

    # only for test_test.(test_test는 사실 test_train 케이스도 포함임. 그래서 test_train인 경우 = w,b모두 None인 경우, 그냥 로버트 모델을 로드해서 씀)
    def loadWeight(self, vae_model, w=None, b=None):
        self.model_train.load_weights(vae_model)
        print("loaded weight from robert : ", self.model_train.get_weights()[58], self.model_train.get_weights()[59])
        if (len(w) == len(b) == 12):
            # w and b are from 'train_test' model. Thus, they are just for one au. ==> Need to magnify.
            w = np.array(w)
            b = np.array(b)
            print('>>>>>>>>>>>>>>> w shape: ', w.shape)
            print('>>>>>>>>>>>>>>> b shape: ', b.shape)
            print('-----------------------------')
            w = np.transpose(w, (1, 0, 2))  # (12,2000,2) -> (2000,12,2)
            b = np.transpose(b, (1, 0, 2))
            print('>>>>>>>>>>>>>>> w shape: ', w.shape)
            print('>>>>>>>>>>>>>>> b shape: ', b.shape)
            self.model_train.layers[-1].weights[0].load(w)
            self.model_train.layers[-1].weights[1].load(b)
            print("loaded weight from maml : ", self.model_train.get_weights()[58], self.model_train.get_weights()[59])

    # only for test_test. 로드한 weight으로 pred값 도출. 배치로 한방에 predict하기 위해 로버트 모델을 쓴것.
    def testWithSavedModel(self, x):
        z, pred = self.model_z_int.predict(x, batch_size=len(x))
        return pred
