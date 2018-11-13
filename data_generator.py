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
            self.metatrain_character_folders = []
            all_aus = os.listdir(data_folder)
            num_subjects = int(FLAGS.meta_batch_size / len(all_aus))
            print('FOR M0 - num_subjects in one au: ', FLAGS.meta_batch_size / len(all_aus))
            for au in all_aus:
                subjects = os.listdir(os.path.join(data_folder, au))
                subjects.sort()
                subject_folders = [os.path.join(data_folder, au, subject) for subject in subjects]
                self.metatrain_character_folders.extend(subject_folders[
                                                        FLAGS.sbjt_start_idx:FLAGS.sbjt_start_idx + num_subjects])


    def make_data_tensor(self, train=True):
        folders = self.metatrain_character_folders
        print(">>>>>>> train folders: ", folders)

        # make list of files
        print('Generating filenames')
        inputa_features = []
        inputb_features = []
        labelas = []
        labelbs = []
        # To have totally different inputa and inputb, they should be sampled at the same time and then split.
        for sub_folder in folders:
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

        print("--------------------------------------------")
        print(">>>>>>>>>>>>>>>>> vae_model: ", FLAGS.vae_model)
        print(">>>>>>>>>>>>>>>>>> random seed for kshot: ", FLAGS.kshot_seed)
        print(">>>>>>>>>>>>>>>>>> random seed for weight: ", FLAGS.weight_seed)

        #################################################################################

        three_layers = feature_layer(10, FLAGS.num_au)
        three_layers.loadWeight(FLAGS.vae_model, FLAGS.au_idx, num_au_for_rm=FLAGS.num_au)

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
