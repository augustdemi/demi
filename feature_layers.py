import EmoEstimator as EE
from build_vae import VAE
from keras.layers import Dense, Input, Reshape
import keras as K
class feature_layer:
    def __init__(self, batch_size, num_au):  # num_au : will be used for building soft max layer

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

    # 이미 만들어진 vae로 부터 3개 레이어에 대한 weight만 취해옴
    def loadWeight(self, vae_model_name, au_index, num_au_for_rm=1, w=None, b=None):
        if num_au_for_rm > 1:
            trained_model = VAE((160, 240, 1), self.batch_size, num_au_for_rm).model_train
            print(">>>>>>>>> model loaded from ", vae_model_name)
            trained_model.load_weights(vae_model_name + '.h5')
            #### get weight
            layer_dict_whole_vae = dict([(layer.name, layer) for layer in trained_model.layers])
            w_intermediate = layer_dict_whole_vae['intermediate'].get_weights()
            w_z_mean = layer_dict_whole_vae['z_mean'].get_weights()
            print('check the last layer of model_train: ', trained_model.layers[-1].name)
            import pickle
            out = open("/home/ml1323/project/robert_code/vae_log/leaveoo/itermediate_layers.pkl", 'wb')
            weights_to_save = {'w_intermediate': w_intermediate, 'w_z_mean': w_z_mean}
            pickle.dump(weights_to_save, out, protocol=2)
            out.close()
            w_softmaxpdf_1 = trained_model.layers[-1].get_weights()
            # print("[vae_model]loaded weight from VAE : ", w_softmaxpdf_1[1])

            # whene w and b is not None = w and b is from MAML
            if w is not None and b is not None:
                w_softmaxpdf_1 = [w, b]
                print("[vae_model]loaded weight from MAML : ", w_softmaxpdf_1[1])

            #### set weight for 3 layers
            layer_dict_3layers = dict([(layer.name, layer) for layer in self.model_intensity.layers])
            layer_dict_3layers['intermediate'].set_weights(w_intermediate)
            layer_dict_3layers['z_mean'].set_weights(w_z_mean)
            print('check the last layer of model_intensity: ', self.model_intensity.layers[-1].name)

            if w_softmaxpdf_1[1].shape[0] == self.num_au:
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
        else:
            self.model_intensity.load_weights(vae_model_name + '.h5')
            # whene w and b is not None = w and b is from MAML
            if w is not None and b is not None:
                w_softmaxpdf_1 = [w, b]
                print("[vae_model] loaded weight from MAML : ", w_softmaxpdf_1[1])
                self.model_intensity.layers[-1].set_weights(w_softmaxpdf_1)
            layer_dict_3layers = dict([(layer.name, layer) for layer in self.model_intensity.layers])
            print("[vae_model] final loaded weight : ")
            print("[vae_model] b1 : ", layer_dict_3layers['intermediate'].get_weights()[1])
            print("[vae_model] b2 : ", layer_dict_3layers['z_mean'].get_weights()[1])
            print("[vae_model] b3 : ", self.model_intensity.layers[-1].get_weights()[1])

    def loadWeightS(self, vae_model_name, au_index, num_au_for_rm=1, w=None, b=None):
        if num_au_for_rm > 1:
            trained_model = VAE((160, 240, 1), self.batch_size, num_au_for_rm).model_train
            print(">>>>>>>>> model loaded from ", vae_model_name)
            trained_model.load_weights(vae_model_name + '.h5')
            #### get weight
            layer_dict_whole_vae = dict([(layer.name, layer) for layer in trained_model.layers])
            w_intermediate = layer_dict_whole_vae['intermediate'].get_weights()
            w_z_mean = layer_dict_whole_vae['z_mean'].get_weights()
            print('check the last layer of model_train: ', trained_model.layers[-1].name)
            w_softmaxpdf_1 = trained_model.layers[-1].get_weights()
            # print("[vae_model]loaded weight from VAE : ", w_softmaxpdf_1[1])

            # whene w and b is not None = w and b is from MAML
            if w is not None and b is not None:
                w_softmaxpdf_1 = [w, b]
                print("[vae_model]loaded weight from MAML : ", w_softmaxpdf_1[1])

            #### set weight for 3 layers
            layer_dict_3layers = dict([(layer.name, layer) for layer in self.model_intensity.layers])
            layer_dict_3layers['intermediate'].set_weights(w_intermediate)
            layer_dict_3layers['z_mean'].set_weights(w_z_mean)
            print('check the last layer of model_intensity: ', self.model_intensity.layers[-1].name)
        else:
            self.model_intensity.load_weights(vae_model_name + '.h5')
            # whene w and b is not None = w and b is from MAML
            if w is not None and b is not None:
                w_softmaxpdf_1 = [w, b]
                print("[vae_model] loaded weight from MAML : ", w_softmaxpdf_1[1])
                self.model_intensity.layers[-1].set_weights(w_softmaxpdf_1)
        print("[vae_model] final loaded weight : ")
        print("[vae_model] b1 : ", layer_dict_3layers['intermediate'].get_weights()[1])
        print("[vae_model] b2 : ", layer_dict_3layers['z_mean'].get_weights()[1])
        print("[vae_model] b3 : ", self.model_intensity.layers[-1].get_weights()[1])


    def loadWeight_m0(self, vae_model_name, w, b, au_index=-1):

        trained_model = VAE((160, 240, 1), self.batch_size, self.TOTAL_AU).model_train
        print(">>>>>>>>> model loaded from ", vae_model_name)
        trained_model.load_weights(vae_model_name + '.h5')
        #### get weight
        layer_dict_whole_vae = dict([(layer.name, layer) for layer in trained_model.layers])
        w_intermediate = layer_dict_whole_vae['intermediate'].get_weights()
        w_z_mean = layer_dict_whole_vae['z_mean'].get_weights()
        print('check the last layer of model_train: ', trained_model.layers[-1].name)
        w_softmaxpdf_1 = []
        print("[vae_model]loaded weight from VAE : ", w_softmaxpdf_1)
        import numpy as np
        # whene w and b is not None = w and b is from MAML
        w_arr = w
        b_arr = b
        for i in range(7):
            w_arr = np.hstack((w_arr, w))
            b_arr = np.vstack((b_arr, b))
        w_softmaxpdf_1 = [w_arr, b_arr]
        print("[vae_model] w shape from MAML : ", w_softmaxpdf_1[0].shape)
        print("[vae_model] b shape from MAML : ", w_softmaxpdf_1[1].shape)
        print("[vae_model]loaded weight from MAML : ", w_softmaxpdf_1[1])

        #### set weight for 3 layers
        layer_dict_3layers = dict([(layer.name, layer) for layer in self.model_intensity.layers])
        layer_dict_3layers['intermediate'].set_weights(w_intermediate)
        layer_dict_3layers['z_mean'].set_weights(w_z_mean)
        print('check the last layer of model_intensity: ', self.model_intensity.layers[-1].name)

        self.model_intensity.layers[-1].set_weights(w_softmaxpdf_1)

    def loadWeight_from_maml(self, weights, au_index):
        #### set weight for 3 layers
        layer_dict_3layers = dict([(layer.name, layer) for layer in self.model_intensity.layers])
        layer_dict_3layers['intermediate'].set_weights([weights['w1'], weights['b1']])
        layer_dict_3layers['z_mean'].set_weights([weights['w2'], weights['b2']])
        print('check the last layer of model_intensity: ', self.model_intensity.layers[-1].name)

        if weights['b3'].shape[0] == self.num_au:
            self.model_intensity.layers[-1].set_weights([weights['w3'], weights['b3']])
        else:
            print(">>>>>>>>>>> going to choose this index in VAE:", au_index)
            try:
                w = weights['w3'][:, au_index]
                b = weights['b3'][au_index]
                w = w.reshape(self.latent_dim3, 1, 2)
                b = b.reshape(1, 2)
                self.model_intensity.layers[-1].set_weights([w, b])
            except IndexError as err:
                print("###########################IndexError:", err)

        print("[vae_model] final loaded weight : ")
        print("[vae_model] b1 : ", layer_dict_3layers['intermediate'].get_weights()[1])
        print("[vae_model] b2 : ", layer_dict_3layers['z_mean'].get_weights()[1])
        print("[vae_model] b3 : ", self.model_intensity.layers[-1].get_weights()[1])
