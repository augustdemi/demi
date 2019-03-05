import EmoEstimator as EE
from build_vae import VAE
from keras.layers import Dense, Input, Reshape
import keras as K


class feature_layer:
    def __init__(self, batch_size, num_au):  # num_au : will be used for building soft max layer


        latent_dim2 = 500
        latent_dim3 = 300
        num_of_intensity = 2
        TOTAL_AU = 8

        ################# From here, reconstruct the model from input = 2048 with only 3 required layers to finetune only softmax layer
        inp_1 = Input(shape=[latent_dim2])
        z_mean = Dense(latent_dim3, name='z_mean')(inp_1)  # into latent_dim = 300은. output space의 dim이 될것.
        out_1 = EE.layers.softmaxPDF(num_au, num_of_intensity)(Reshape((latent_dim3, 1))(z_mean))

        model_intensity = K.models.Model([inp_1], [out_1])
        model_final_latent_feat = K.models.Model([inp_1], [z_mean])

        self.model_final_latent_feat = model_final_latent_feat
        self.model_intensity = model_intensity
        self.TOTAL_AU = TOTAL_AU
        self.num_au = num_au
        self.batch_size = batch_size
        self.latent_dim3 = latent_dim3

    # 이미 만들어진 vae로 부터 3개 레이어에 대한 weight만 취해옴
    def loadWeight(self, vae_model_name, au_index=-1, w=None, b=None):
        trained_model = VAE((160, 240, 1), self.batch_size, self.num_au).model_train
        print(">>>>>>>>> model loaded from ", vae_model_name)
        trained_model.load_weights(vae_model_name + '.h5')

        layer_dict_whole_vae = dict([(layer.name, layer) for layer in trained_model.layers])
        w_z_mean = layer_dict_whole_vae['z_mean'].get_weights()
        print('check the last layer of model_train: ', trained_model.layers[-1].name)
        w_softmaxpdf_1 = trained_model.layers[-1].get_weights()
        # whene w and b is not None = w and b is from MAML
        if w is not None and b is not None:
            w_softmaxpdf_1 = [w, b]
            print("[vae_model]loaded weight from MAML : ", w_softmaxpdf_1[1])

        print('check the last layer of model_intensity: ', self.model_intensity.layers[-1].name)
        #### set weight for 3 layers
        layer_dict_3layers = dict([(layer.name, layer) for layer in self.model_intensity.layers])
        layer_dict_3layers['z_mean'].set_weights(w_z_mean)
        if au_index < 0:
            self.model_intensity.layers[-1].set_weights(w_softmaxpdf_1)
        else:
            print(">>>>>>>>>>> going to choose this index in VAE:", au_index)
            try:
                w = w_softmaxpdf_1[0][:, au_index]
                b = w_softmaxpdf_1[1][au_index]
                w = w.reshape(self.latent_dim3, 1, 2)
                b = b.reshape(1, 2)
                self.model_intensity.layers[-1].set_weights([w, b])
            except IndexError as err:
                print("###########################IndexError:", err)
