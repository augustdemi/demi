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
        subjects = subjects[FLAGS.sbjt_start_idx:FLAGS.sbjt_start_idx + FLAGS.num_test_task]
        print('>>>>>>>>>>>>>> selected subjects from feat_vec: ', subjects)
        self.feature_files = [os.path.join(data_folder, subject) for subject in subjects]

        label_folder = FLAGS.labeldir  # label_folder = '/home/ml1323/project/robert_data/DISFA/label/'
        subjects = os.listdir(label_folder)
        subjects.sort()
        subjects = subjects[FLAGS.sbjt_start_idx:FLAGS.sbjt_start_idx + FLAGS.num_test_task]
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
        all_used_frame_set = []
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
                test_subjects = ['SN017', 'SN018', 'SN021', 'SN023', 'SN024', 'SN025', 'SN026', 'SN027', 'SN028',
                                 'SN029', 'SN030',
                                 'SN031', 'SN032']
                import pickle
                data_source = 'test' if FLAGS.adaptation else 'train'
                save_path = '/home/ml1323/project/robert_code/new/check_labels/' + data_source + '/' + str(
                    FLAGS.update_batch_size) + 'shot'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                if FLAGS.adaptation:
                    save_path = os.path.join(save_path, test_subjects[FLAGS.sbjt_start_idx] + '_' + au + ".pkl")
                else:
                    save_path = os.path.join(save_path, au + ".pkl")
                out = open(save_path, 'wb')
                pickle.dump({'off': selected_off_frame_idx,
                             'on': selected_on_frame_idx}, out, protocol=2)

            for i in range(FLAGS.meta_batch_size):
                # split selected idx into two for inputa / inputb
                half_off_frame = int(len(selected_off_frame_idx[i]) / 2)
                half_on_frame = int(len(selected_on_frame_idx[i]) / 2)
                inputa_idx = selected_off_frame_idx[i][:half_off_frame]
                inputa_idx.extend(selected_on_frame_idx[i][:half_on_frame])
                inputa.append(self.feat_vec[i][inputa_idx])
                labela.append(self.labels[i][inputa_idx])
                # select inputa / inputb
                inputb_idx = selected_off_frame_idx[i][half_off_frame:]
                inputb_idx.extend(selected_on_frame_idx[i][half_on_frame:])
                inputb.append(self.feat_vec[i][inputb_idx])
                labelb.append(self.labels[i][inputb_idx])

                if FLAGS.evaluate:
                    all_used_frame_set.extend(selected_on_frame_idx[0])
                    all_used_frame_set.extend(selected_off_frame_idx[0])
        inputa = np.array(inputa)
        inputb = np.array(inputb)
        labela = np.array(labela)
        labelb = np.array(labelb)
        all_used_frame_set = list(set(all_used_frame_set))
        return inputa, inputb, labela, labelb, all_used_frame_set

    def same_random_data(self, seed, kshot, aus):
        inputa = []
        labela = []

        frames_to_select = random.sample(range(len(self.feat_vec[0])), kshot)
        for _ in aus:
            inputa.append(self.feat_vec[0][frames_to_select])
            labela.append(self.labels[0][frames_to_select])
        inputa = np.array(inputa)
        inputb = np.array(inputa)
        labela = np.array(labela)
        labelb = np.array(labela)

        print(frames_to_select)
        print(labela)

        if FLAGS.check_sample:
            test_subjects = ['SN017', 'SN018', 'SN021', 'SN023', 'SN024', 'SN025', 'SN026', 'SN027', 'SN028',
                             'SN029', 'SN030',
                             'SN031', 'SN032']
            import pickle
            save_path = '/home/ml1323/project/robert_code/new/check_labels/test/' + str(
                FLAGS.update_batch_size) + 'shot_same_random'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_path = os.path.join(save_path, test_subjects[FLAGS.sbjt_start_idx] + ".pkl")
            out = open(save_path, 'wb')
            pickle.dump(frames_to_select, out, protocol=2)

        return inputa, inputb, labela, labelb, frames_to_select

    def sample_test_data(self, seed, kshot, aus, subject_idx):
        inputa = []
        labela = []
        all_used_frame_set = []
        for au in aus:
            print('==== au: ', au)
            one_au_one_subject_on_frame_indices = self.on_info_df[au][subject_idx]
            random.seed(seed)
            selected_on_frame_idx = random.sample(one_au_one_subject_on_frame_indices,
                                                  min(kshot, int(len(one_au_one_subject_on_frame_indices) / 2)))
            print('-- selected_on_frame_idx: ', selected_on_frame_idx)

            one_au_one_subject_off_frame_indices = self.off_info_df[au][subject_idx]
            needed_num_samples = 2 * kshot - len(selected_on_frame_idx)
            random.seed(seed)
            selected_off_frame_idx = random.sample(one_au_one_subject_off_frame_indices, needed_num_samples)
            print('-- selected_off_frame_idx: ', selected_off_frame_idx)

            if FLAGS.check_sample:
                import pickle
                save_path = '/home/ml1323/project/robert_code/new/check_labels/test/' + str(
                    FLAGS.update_batch_size) + 'shot'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                test_subjects = ['SN017', 'SN018', 'SN021', 'SN023', 'SN024', 'SN025', 'SN026', 'SN027', 'SN028',
                                 'SN029', 'SN030',
                                 'SN031', 'SN032']
                save_path = os.path.join(save_path, test_subjects[FLAGS.sbjt_start_idx] + '_' + au + ".pkl")
                out = open(save_path, 'wb')
                pickle.dump({'off': selected_off_frame_idx,
                             'on': selected_on_frame_idx}, out, protocol=2)

            inputa.append(self.feat_vec[subject_idx][selected_off_frame_idx])
            labela.append(self.labels[subject_idx][selected_off_frame_idx])
            if FLAGS.evaluate:
                all_used_frame_set.extend(selected_on_frame_idx)
                all_used_frame_set.extend(selected_off_frame_idx)
        inputa = np.array(inputa)
        labela = np.array(labela)
        all_used_frame_set = list(set(all_used_frame_set))
        return inputa, inputa, labela, labela, all_used_frame_set
