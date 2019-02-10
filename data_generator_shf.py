""" Code for loading data. """
import numpy as np
import os
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import get_all_feature_w_all_labels
from feature_layers import feature_layer
import random

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
        self.total_num_au = 8

        data_folder = FLAGS.datadir
        subjects = os.listdir(data_folder)
        subjects.sort()
        subjects = subjects[FLAGS.sbjt_start_idx:FLAGS.sbjt_start_idx + FLAGS.meta_batch_size]
        print('>>>>>>>>>>>>>> selected subjects from feat_vec: ', subjects)
        self.feature_files = [os.path.join(data_folder, subject) for subject in subjects]

        label_folder = FLAGS.labeldir  # label_folder = '/home/ml1323/project/robert_data/DISFA/label/'
        subjects = os.listdir(label_folder)
        subjects.sort()
        subjects = subjects[FLAGS.sbjt_start_idx:FLAGS.sbjt_start_idx + FLAGS.meta_batch_size]
        self.label_folder = [os.path.join(label_folder, subject) for subject in subjects]

    def make_data_tensor(self):
        feat_vec, labels, on_info_df, off_info_df = get_all_feature_w_all_labels(self.feature_files, self.label_folder)

        ################################### dim reduction ####################################
        print(">>>>>>>>>>>>>>>>> embedding model: ", FLAGS.vae_model)
        three_layers = feature_layer(10, FLAGS.num_au)
        three_layers.loadWeight(FLAGS.vae_model, FLAGS.au_idx, num_au_for_rm=FLAGS.num_au)
        feat_vec = three_layers.model_final_latent_feat.predict(feat_vec)
        print(">>> z_arr len:", len(feat_vec))

        #################################### make tensor ###############################
        self.feat_tensor = feat_tensor = tf.convert_to_tensor(feat_vec)
        self.label_tensor = label_tensor = tf.convert_to_tensor(labels)
        self.on_info_df = on_info_df
        self.off_info_df = off_info_df

    def shuffle_data_tensor(self, seed, aus, kshot):
        inputa = []
        inputb = []
        labela = []
        labelb = []
        for au in aus:
            print('==================== au: ', au)
            one_au_all_subjects_on_frame_indices = self.on_info_df[au]
            selected_on_frame_idx = []
            for each_subj_idx in one_au_all_subjects_on_frame_indices:
                random.seed(seed)
                selected_on_frame_idx.append(random.sample(each_subj_idx, min(2 * kshot, len(each_subj_idx))))
            print('>>> selected_on_frame_idx: ', selected_on_frame_idx)
            one_au_all_subjects_off_frame_indices = self.off_info_df[au]
            selected_off_frame_idx = []
            for i in range(len(one_au_all_subjects_off_frame_indices)):
                each_subj_idx = one_au_all_subjects_off_frame_indices[i]
                needed_num_samples = 4 * kshot - len(selected_on_frame_idx[i])
                random.seed(seed)
                selected_off_frame_idx.append(random.sample(each_subj_idx, needed_num_samples))
            print('>>> selected_off_frame_idx: ', selected_off_frame_idx)
            for i in range(FLAGS.meta_batch_size):
                inputa.append(tf.gather(self.feat_tensor[i], selected_off_frame_idx[i][:kshot]))
                inputa.append(tf.gather(self.feat_tensor[i], selected_on_frame_idx[i][:kshot]))
                inputb.append(tf.gather(self.feat_tensor[i], selected_off_frame_idx[i][kshot:]))
                inputb.append(tf.gather(self.feat_tensor[i], selected_on_frame_idx[i][kshot:]))
                labela.append(tf.gather(self.label_tensor[i], selected_off_frame_idx[i][:kshot]))
                labela.append(tf.gather(self.label_tensor[i], selected_on_frame_idx[i][:kshot]))
                labelb.append(tf.gather(self.label_tensor[i], selected_off_frame_idx[i][kshot:]))
                labelb.append(tf.gather(self.label_tensor[i], selected_on_frame_idx[i][kshot:]))
        inputa = tf.concat(inputa, 0)
        inputb = tf.concat(inputb, 0)
        labela = tf.concat(labela, 0)
        labelb = tf.concat(labelb, 0)
        return inputa, inputb, labela, labelb
