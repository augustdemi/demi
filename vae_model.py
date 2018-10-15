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
        print("shape before flatten: ", shape)
        print("shape after flatten: ", emb.get_shape())
        n_feat = prod(shape)

        emb = Dropout(0.5)(emb)

        latent_feat = Dense(latent_dim1, activation='relu', name='latent_feat')(emb)  # into 2048
        intermediate = Dense(latent_dim2, activation='relu', name='intermediate')(latent_feat)  # into 500
        z_mean = Dense(latent_dim3, name='z_mean')(intermediate)  # into latent_dim = 300은. output space의 dim이 될것.
        z_log_sigma = Dense(latent_dim3)(intermediate)
        print('==============================')
        print('emb', emb.shape)
        print('latent_feat', latent_feat.shape)
        print('intermediate', intermediate.shape)
        print('z_mean', z_mean.shape)
        print('z_log_sigma', z_log_sigma.shape)
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

        print('z', z.shape)
        print('h_decoded1', h_decoded1.shape)
        print('h_decoded2', h_decoded2.shape)
        print('x_decoded_mean', x_decoded_mean.shape)
        print('==============================')

        out_0 = EE.networks.decoder(x_decoded_mean, shape, norm=1)

        model_train = K.models.Model([inp_0], [out_0, out_1, out_0])
        model_deep_feature = K.models.Model([inp_0], [latent_feat])

        ################# Above: to load the model.
        ################# From here, reconstruct the model from input = 2048 with only 3 required layers
        inp_1 = Input(shape=[latent_dim1])
        intermediate = Dense(latent_dim2, activation='relu', name='intermediate')(inp_1)  # into 500
        z_mean = Dense(latent_dim3, name='z_mean')(intermediate)  # into latent_dim = 300은. output space의 dim이 될것.
        out_1 = EE.layers.softmaxPDF(num_au, num_of_intensity)(Reshape((latent_dim3, 1))(z_mean))
        model_z_intensity = K.models.Model([inp_1], [z_mean, out_1])


        self.model_train = model_train
        self.model_z_intensity = model_z_intensity
        self.model_deep_feature = model_deep_feature

    def loadWeight(self, vae_model, w=None, b=None, iterative_au=False):
        self.model_train.load_weights(vae_model)
        # get weight
        layer_dict_whole_vae = dict([(layer.name, layer) for layer in self.model_train.layers])
        w_intermediate = layer_dict_whole_vae['intermediate'].get_weights()
        w_z_mean = layer_dict_whole_vae['z_mean'].get_weights()
        w_softmaxpdf_1 = layer_dict_whole_vae['softmaxpdf_1'].get_weights()
        print("[vae_model]loaded weight from VAE : ", w_softmaxpdf_1)

        # whene w and b is not None = w and b is from MAML
        if w is not None and b is not None:
            w_softmaxpdf_1 = [w, b]
            print("[vae_model]loaded weight from MAML : ", w_softmaxpdf_1)

        # set weight for 3 layers
        layer_dict_3layers = dict([(layer.name, layer) for layer in self.model_z_intensity.layers])
        layer_dict_3layers['intermediate'].set_weights(w_intermediate)
        layer_dict_3layers['z_mean'].set_weights(w_z_mean)
        layer_dict_3layers['softmaxpdf_1'].set_weights(w_softmaxpdf_1)






    def computeLatentVal(self, x, vae_model, au_idx):
        if vae_model.endswith('h5'):
            self.model_train.load_weights(vae_model)
        else:
            print('base vae in interative case: ', vae_model + '/' + os.listdir(vae_model)[0])
            self.model_train.load_weights(vae_model + '/' + os.listdir(vae_model)[0])
        z, _ = self.model_z_intensity.predict(x, batch_size=len(x))
        print(">>>>>>>>> z shape:", z.shape)
        layer_dict_whole_vae = dict([(layer.name, layer) for layer in self.model_train.layers])
        loaded_weight = layer_dict_whole_vae['softmaxpdf_1'].get_weights()

        print('[vae_model]shape of loaded_weight in computeLatentVal(): ', loaded_weight[0].shape,
              loaded_weight[1].shape)
        if (loaded_weight[1].shape[0] > 1) and (au_idx < 12):
            w = loaded_weight[0][:, au_idx]
            b = loaded_weight[1][au_idx]
            loaded_weight = [w.reshape(2000, 1, 2), b.reshape(1, 2)]
            print('[vae_model]after: shape of loaded_weight in computeLatentVal(): ', loaded_weight[0].shape,
                  loaded_weight[1].shape)
        return loaded_weight, z

    # only for test_test.(test_test는 사실 test_train 케이스도 포함임. 그래서 test_train인 경우 = w,b모두 None인 경우, 그냥 로버트 모델을 로드해서 씀)
    def loadWeight_prev(self, vae_model, w=None, b=None, iterative_au=False):
        if iterative_au:
            print("######## dir for iterative load of model: ", vae_model)
            temp_vae_model = VAE((160, 240, 1), 32, 1)
            w_arr = None
            b_arr = None
            for i in range(12):
                temp_vae_model.model_train.load_weights(vae_model + '/au' + str(i) + '.h5')
                for j in range(len(self.model_train.layers) - 1):
                    loaded = temp_vae_model.model_train.layers[j].get_weights()
                    self.model_train.layers[j].set_weights(loaded)
                w = temp_vae_model.model_train.get_weights()[58]
                b = temp_vae_model.model_train.get_weights()[59]
                w = w.reshape(2000, 1, 2)
                b = b.reshape(1, 2)
                if w_arr is None:
                    w_arr = w
                    b_arr = b
                else:
                    w_arr = np.hstack((w_arr, w))
                    b_arr = np.vstack((b_arr, b))
            print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
            print('w shape: ', w_arr.shape)
            print('b shape: ', b_arr.shape)

            self.model_train.layers[-1].set_weights([w_arr, b_arr])
            print("[vae_model]loaded weight from robert : ", self.model_train.get_weights()[58],
                  self.model_train.get_weights()[59])
        else:
            self.model_train.load_weights(vae_model)
            print("[vae_model]loaded weight from robert : ", self.model_train.get_weights()[58],
                  self.model_train.get_weights()[59])
            print("And shape of w: ", self.model_train.get_weights()[58].shape)
            if w is not None and b is not None:
                self.model_train.layers[-1].weights[0].load(w)
                self.model_train.layers[-1].weights[1].load(b)
                print("[vae_model]loaded weight from maml : ", self.model_train.get_weights()[58],
                      self.model_train.get_weights()[59])

    # only for test_test. 로드한 weight으로 pred값 도출. 배치로 한방에 predict하기 위해 로버트 모델을 쓴것.
    def testWithSavedModel(self, x):
        _, pred = self.model_z_intensity.predict(x, batch_size=len(x))
        return pred
