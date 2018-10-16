import EmoEstimator as EE
from build_vae import VAE
from keras.layers import Dense, Lambda, Input, Reshape
import keras as K
import numpy as np
import os


class feature_layer:
    def __init__(self, batch_size, num_au):

        latent_dim1 = 2048
        latent_dim2 = 500
        latent_dim3 = 300
        num_of_intensity = 2
        TOTAL_AU = 8

        ################# From here, reconstruct the model from input = 2048 with only 3 required layers to finetune only softmax layer
        inp_1 = Input(shape=[latent_dim1])
        intermediate = Dense(latent_dim2, activation='relu', name='intermediate')(inp_1)  # into 500
        z_mean = Dense(latent_dim3, name='z_mean')(intermediate)  # into latent_dim = 300은. output space의 dim이 될것.
        out_1 = EE.layers.softmaxPDF(num_au, num_of_intensity)(Reshape((latent_dim3, 1))(z_mean))

        model_intensity = K.models.Model([inp_1], [out_1])
        model_final_latent_feat = K.models.Model([inp_1], [z_mean])

        self.model_final_latent_feat = model_final_latent_feat
        self.model_intensity = model_intensity
        self.TOTAL_AU = TOTAL_AU
        self.num_au = num_au
        self.batch_size = batch_size
        self.latent_dim3 = latent_dim3

    # kdt
    def loadWeight(self, vae_model_name, w=None, b=None, au_index=-1):

        trained_model = VAE((160, 240, 1), self.batch_size, self.TOTAL_AU).model_train
        print(">>>>>>>>> model loaded from ", vae_model_name)
        trained_model.load_weights(vae_model_name + '.h5')
        # get weight
        layer_dict_whole_vae = dict([(layer.name, layer) for layer in trained_model.layers])
        w_intermediate = layer_dict_whole_vae['intermediate'].get_weights()
        w_z_mean = layer_dict_whole_vae['z_mean'].get_weights()
        print('check the last layer of model_train: ', trained_model.layers[-1].name)
        w_softmaxpdf_1 = trained_model.layers[-1].get_weights()
        print("[vae_model]loaded weight from VAE : ", w_softmaxpdf_1)

        # whene w and b is not None = w and b is from MAML
        if w is not None and b is not None:
            w_softmaxpdf_1 = [w, b]
            print("[vae_model]loaded weight from MAML : ", w_softmaxpdf_1)

        # set weight for 3 layers
        layer_dict_3layers = dict([(layer.name, layer) for layer in self.model_intensity.layers])
        layer_dict_3layers['intermediate'].set_weights(w_intermediate)
        layer_dict_3layers['z_mean'].set_weights(w_z_mean)
        print('check the last layer of model_intensity: ', self.model_intensity.layers[-1].name)
        trained_model.summary()
        self.model_intensity.summary()

        if w_softmaxpdf_1[1].shape[0] == self.num_au:
            self.model_intensity.layers[-1].set_weights(w_softmaxpdf_1)
        else:
            w = w_softmaxpdf_1[0][:, au_index]
            b = w_softmaxpdf_1[1][au_index]
            w = w.reshape(self.latent_dim3, 1, 2)
            b = b.reshape(1, 2)
            self.model_intensity.layers[-1].set_weights([w, b])

    def computeLatentVal(self, x, vae_model, au_idx):
        if vae_model.endswith('h5'):
            self.model_train.load_weights(vae_model)
        else:
            print('base vae in interative case: ', vae_model + '/' + os.listdir(vae_model)[0])
            self.model_train.load_weights(vae_model + '/' + os.listdir(vae_model)[0])
        ##############################
        z, _ = self.model_z_intensity.predict(x, batch_size=len(x))
        print(">>>>>>>>> z shape:", z.shape)
        layer_dict_whole_vae = dict([(layer.name, layer) for layer in self.model_train.layers])
        loaded_weight = layer_dict_whole_vae['softmaxpdf_1'].get_weights()
        ##############################
        print('[vae_model]shape of loaded_weight in computeLatentVal(): ', loaded_weight[0].shape,
              loaded_weight[1].shape)
        if (loaded_weight[1].shape[0] > 1) and (au_idx < self.TOTAL_AU):
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
            for i in range(self.TOTAL_AU):
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
