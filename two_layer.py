import EmoEstimator as EE
from build_vae import VAE
from keras.layers import Dense, Input, Reshape, Lambda, Activation
import keras as K
import numpy as np
import keras.backend as KB

class feature_layer:
    def __init__(self, batch_size, num_au):  # num_au : will be used for building soft max layer

        latent_dim2 = 500
        latent_dim3 = 300
        num_of_intensity = 2

        TOTAL_AU = 8

        ################# From here, reconstruct the model from input = 2048 with only 3 required layers to finetune only softmax layer
        inp_1 = Input(shape=[latent_dim2])
        intermediate_relu = Activation(activation='relu', name='intermediate_relu')(inp_1)  # into 500
        z_mean = Dense(latent_dim3, name='z_mean')(intermediate_relu)
        z_log_sigma = Dense(latent_dim3,  name='z_sig')(intermediate_relu)
        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = []
            for m, s in zip(np.zeros(latent_dim3), np.ones(latent_dim3)):
                epsilon.append(KB.random_normal(shape=[batch_size, 1], mean=m, std=s))
            epsilon = KB.concatenate(epsilon, 1)
            return z_mean + KB.exp(z_log_sigma) * epsilon

        z = Lambda(sampling, output_shape=(latent_dim3,))([z_mean, z_log_sigma])
        out_1 = EE.layers.softmaxPDF(num_au, num_of_intensity)(Reshape((latent_dim3, 1))(z))

        h_decoded1 = Dense(latent_dim2, name='to500')(z)  # into 500



        model_intensity = K.models.Model([inp_1], [out_1])
        model_reconst = K.models.Model([inp_1], [h_decoded1])
        model_final_latent_feat = K.models.Model([inp_1], [z])

        trained_model = VAE((160, 240, 1), batch_size, num_au).model_train
        self.trained_model = trained_model
        self.load_flag = False

        self.model_final_latent_feat = model_final_latent_feat
        self.model_intensity = model_intensity
        self.model_reconst = model_reconst
        self.TOTAL_AU = TOTAL_AU
        self.num_au = num_au
        self.batch_size = batch_size
        self.latent_dim3 = latent_dim3

    def loadWeight(self, vae_model_name, au_index=-1, w=None, b=None):
        print(">>>>>>>>> model loaded from ", vae_model_name)

        if not self.load_flag:
            self.trained_model.load_weights(vae_model_name + '.h5')

        layer_dict_whole_vae = dict([(layer.name, layer) for layer in self.trained_model.layers])
        w_z_mean = layer_dict_whole_vae['z_mean'].get_weights()
        w_z_sig = layer_dict_whole_vae['z_sig'].get_weights()
        w_to500 = layer_dict_whole_vae['to500'].get_weights()
        print('check the last layer of trained_model: ', self.trained_model.layers[-1].name)
        w_softmaxpdf_1 = self.trained_model.layers[-1].get_weights()
        print("[two_layer]loaded bias of soft from VAE : ", w_softmaxpdf_1[1])
        print("[two_layer]loaded bias of z_mean from VAE : ", w_z_mean[1][:4])

        # whene w and b is not None = w and b is from MAML
        if w is not None and b is not None:
            w_softmaxpdf_1 = [w[0], b[0]]
            w_z_mean = [w[1], b[1]]
            w_z_sig = [w[2], b[2]]
            print("[two_layer]loaded bias of soft from MAML : ", w_softmaxpdf_1[1])
            print("[two_layer]loaded bias of z_mean from MAML : ", w_z_mean[1][:4])


        print('--------------------------------------------------------------------')
        print('check the last layer of model_intensity: ', self.model_intensity.layers[-1].name)
        #### set weight for 3 layers
        layer_dict_3layers = dict([(layer.name, layer) for layer in self.model_reconst.layers])
        layer_dict_3layers['z_mean'].set_weights(w_z_mean)
        layer_dict_3layers['z_sig'].set_weights(w_z_sig)
        layer_dict_3layers['to500'].set_weights(w_to500)
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
