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

        feat_vec, labels, on_info_df, off_info_df = get_all_feature_w_all_labels(self.feature_files, self.label_folder)

        self.feat_vec = feat_vec
        self.labels = labels
        self.on_info_df = on_info_df
        self.off_info_df = off_info_df

    def shuffle_data(self, seed, kshot, aus):
        inputa = []
        inputb = []
        labela = []
        labelb = []
        for au in aus:
            # print('==================== au: ', au)
            one_au_all_subjects_on_frame_indices = self.on_info_df[au]
            selected_on_frame_idx = []
            for each_subj_idx in one_au_all_subjects_on_frame_indices:
                random.seed(seed)
                selected_on_frame_idx.append(random.sample(each_subj_idx, 2 * min(kshot, int(len(each_subj_idx) / 2))))
            print('>>> selected_on_frame_idx: ', selected_on_frame_idx)

            one_au_all_subjects_off_frame_indices = self.off_info_df[au]
            selected_off_frame_idx = []
            for i in range(len(one_au_all_subjects_off_frame_indices)):
                each_subj_idx = one_au_all_subjects_off_frame_indices[i]
                needed_num_samples = 4 * kshot - len(selected_on_frame_idx[i])
                random.seed(seed)
                selected_off_frame_idx.append(random.sample(each_subj_idx, needed_num_samples))
            print('>>> selected_off_frame_idx: ', selected_off_frame_idx)

            if FLAGS.check_sample:
                test_subjects = ['SN001', 'SN002', 'SN003', 'SN004', 'SN005', 'SN006', 'SN007', 'SN008', 'SN009',
                                 'SN010', 'SN011',
                                 'SN012', 'SN013', 'SN016']
                import pickle
                data_source = 'test' if FLAGS.train_test else 'train'
                save_path = '/home/ml1323/project/robert_code/new/check_labels/' + data_source + '/' + str(
                    FLAGS.update_batch_size) + 'shot'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                if FLAGS.train_test:
                    save_path = os.path.join(save_path, test_subjects[FLAGS.sbjt_start_idx] + '_' + au + ".pkl")
                else:
                    save_path = os.path.join(save_path, au + ".pkl")
                out = open(save_path, 'wb')
                pickle.dump({'off': selected_off_frame_idx,
                             'on': selected_on_frame_idx}, out, protocol=2)

            for i in range(FLAGS.meta_batch_size):
                # print('-------------------------------------------------------- subject ', i)
                half_off_frame = int(len(selected_off_frame_idx[i]) / 2)
                half_on_frame = int(len(selected_on_frame_idx[i]) / 2)
                inputa_idx = selected_off_frame_idx[i][:half_off_frame]
                inputa_idx.extend(selected_on_frame_idx[i][:half_on_frame])
                inputa.append(self.feat_vec[i][inputa_idx])
                labela.append(self.labels[i][inputa_idx])

                inputb_idx = selected_off_frame_idx[i][half_off_frame:]
                inputb_idx.extend(selected_on_frame_idx[i][half_on_frame:])
                inputb.append(self.feat_vec[i][inputb_idx])
                labelb.append(self.labels[i][inputb_idx])
        inputa = np.array(inputa)
        inputb = np.array(inputb)
        labela = np.array(labela)
        labelb = np.array(labelb)
        return inputa, inputb, labela, labelb
