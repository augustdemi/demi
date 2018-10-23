""" Code for loading data. """
import numpy as np
import os
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import get_kshot_feature
from feature_layers import feature_layer
FLAGS = flags.FLAGS


class DataGenerator(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """

    def __init__(self, config={}):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
        """
        self.num_samples_per_class = FLAGS.update_batch_size * 2
        self.num_classes = FLAGS.num_classes
        self.img_size = config.get('img_size', (160, 240))
        self.dim_input = np.prod(self.img_size)
        self.weight_dim = 300
        data_folder = FLAGS.datadir
        if data_folder.split('/')[-1].startswith('au'):
            subjects = os.listdir(data_folder)
            subjects.sort()
            subject_folders = [os.path.join(data_folder, subject) for subject in subjects]
            self.metatrain_character_folders = subject_folders[
                                               FLAGS.sbjt_start_idx:FLAGS.sbjt_start_idx + FLAGS.meta_batch_size]
        else:
            for au in os.listdir(data_folder):
                subjects = os.listdir(os.path.join(data_folder, au))
                subjects.sort()
                subject_folders = [os.path.join(data_folder, au, subject) for subject in subjects]
                self.metatrain_character_folders = subject_folders[
                                                   FLAGS.sbjt_start_idx:FLAGS.sbjt_start_idx + FLAGS.meta_batch_size]
        if FLAGS.test_set:  # In test, runs only one test task for the entered subject
            self.metatest_character_folders = [subject_folders[FLAGS.subject_idx]]
        else:
            if FLAGS.train_test:  # test task로 다시한번 모델을 retrain할때는 같은 test task중 train에 쓰이지 않은 다른 샘플을 선택하여 validate
                self.metatest_character_folders = self.metatrain_character_folders
            else:
                self.metatest_character_folders = self.metatrain_character_folders


    def make_data_tensor(self, train=True):
        if train:
            print("===================================2")
            folders = self.metatrain_character_folders
            print(">>>>>>> train folders: ", folders)
        else:
            folders = self.metatest_character_folders
            print(">>>>>>> test folders: ", folders)

        # make list of files
        print('Generating filenames')
        inputa_features = []
        inputb_features = []
        labelas = []
        labelbs = []
        # To have totally different inputa and inputb, they should be sampled at the same time and then splitted.
        for sub_folder in folders:  # 쓰일 task수만큼만 경로 만든다. 이 task들이 iteration동안 어차피 반복될거니까
            # random.shuffle(sampled_character_folders)
            if train:
                off_imgs, on_imgs = get_kshot_feature(sub_folder, FLAGS.feature_path, FLAGS.kshot_seed,
                                                      nb_samples=FLAGS.update_batch_size * 2, validate=False)
            else:
                off_imgs, on_imgs = get_kshot_feature(sub_folder, FLAGS.feature_path, FLAGS.kshot_seed,
                                                      nb_samples=FLAGS.update_batch_size * 2, validate=True)
            # Split data into a/b
            half_off_img = int(len(off_imgs) / 2)
            half_on_img = int(len(on_imgs) / 2)
            inputa_this_subj = []
            inputb_this_subj = []
            for i in range(half_off_img):
                inputa_this_subj.append([float(k) for k in off_imgs[2 * i]])
                inputb_this_subj.append([float(k) for k in off_imgs[2 * i + 1]])
            for i in range(half_on_img):
                inputa_this_subj.append([float(k) for k in on_imgs[2 * i]])
                inputb_this_subj.append([float(k) for k in on_imgs[2 * i + 1]])
            labela_this_subj = [0] * half_off_img
            labela_this_subj.extend([1] * half_on_img)
            labelb_this_subj = [0] * half_off_img
            labelb_this_subj.extend([1] * half_on_img)

            np.random.seed(1)
            np.random.shuffle(inputa_this_subj)
            np.random.seed(1)
            np.random.shuffle(labela_this_subj)

            np.random.seed(2)
            np.random.shuffle(inputb_this_subj)
            np.random.seed(2)
            np.random.shuffle(labelb_this_subj)

            inputa_features.extend(inputa_this_subj)
            inputb_features.extend(inputb_this_subj)
            labelas.extend(labela_this_subj)
            labelbs.extend(labelb_this_subj)

        # print(">>>> inputa_features: ", inputa_features[-1])
        # print(">>> labelas: ", labelas)
        print("--------------------------------------------")
        # print(">>>> inputb_features: ", inputb_features[-1])
        # print(">>> labelbs: ", labelbs)
        print(">>>>>>>>>>>>>>>>> vae_model: ", FLAGS.vae_model)
        print(">>>>>>>>>>>>>>>>>> random seed for kshot: ", FLAGS.kshot_seed)
        print(">>>>>>>>>>>>>>>>>> random seed for weight: ", FLAGS.weight_seed)

        #################################################################################



        batch_size = 10
        three_layers = feature_layer(batch_size, FLAGS.num_au)
        three_layers.loadWeight(FLAGS.vae_model, au_index=FLAGS.au_idx)

        inputa_latent_feat = three_layers.model_final_latent_feat.predict(inputa_features)
        inputb_latent_feat = three_layers.model_final_latent_feat.predict(inputb_features)
        print(">>> z_arr len:", len(inputa_latent_feat))
        #################################################################################

        inputa_latent_feat_tensor = tf.convert_to_tensor(inputa_latent_feat)
        print(inputa_latent_feat_tensor.shape)
        inputa_latent_feat_tensor = tf.reshape(inputa_latent_feat_tensor,
                                               [FLAGS.meta_batch_size, FLAGS.update_batch_size * 2, self.weight_dim])
        inputb_latent_feat_tensor = tf.convert_to_tensor(inputb_latent_feat)
        inputb_latent_feat_tensor = tf.reshape(inputb_latent_feat_tensor,
                                               [FLAGS.meta_batch_size, FLAGS.update_batch_size * 2, self.weight_dim])

        labelas_tensor = tf.convert_to_tensor(labelas)
        labelbs_tensor = tf.convert_to_tensor(labelbs)
        labelas_tensor = tf.one_hot(labelas_tensor, self.num_classes)  ## (num_of_tast, 2NK, N)
        labelbs_tensor = tf.one_hot(labelbs_tensor, self.num_classes)  ## (num_of_tast, 2NK, N)
        labelas_tensor = tf.reshape(labelas_tensor, [FLAGS.meta_batch_size, FLAGS.update_batch_size * 2, 2])
        labelbs_tensor = tf.reshape(labelbs_tensor, [FLAGS.meta_batch_size, FLAGS.update_batch_size * 2, 2])

        return inputa_latent_feat_tensor, inputb_latent_feat_tensor, labelas_tensor, labelbs_tensor
