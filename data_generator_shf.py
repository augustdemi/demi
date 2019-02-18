""" Code for loading data. """
import numpy as np
import os
import tensorflow as tf

from tensorflow.python.platform import flags
from utils import get_all_feature_w_all_labels, test
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

        if FLAGS.adaptation:
            self.inputa, self.labela = test(FLAGS.kshot_path, self.feature_files[0], self.label_folder[0],
                                            FLAGS.kshot_seed, subjects[0], FLAGS.update_batch_size)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(self.inputa.shape)
            print(self.labela.shape)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

        else:
            feat_vec, labels, on_info_df, off_info_df, test_b_frame = get_all_feature_w_all_labels(self.feature_files,
                                                                                                   self.label_folder,
                                                                                                   test_split_seed=FLAGS.test_split_seed)

            self.feat_vec = feat_vec
            self.labels = labels
            self.on_info_df = on_info_df
            self.off_info_df = off_info_df
            self.test_b_frame = test_b_frame
            print('========== will be used this test_b ===========', len(test_b_frame))
            print(test_b_frame)

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

        inputa = np.array(inputa)
        inputb = np.array(inputb)
        labela = np.array(labela)
        labelb = np.array(labelb)
        return inputa, inputb, labela, labelb


    def sample_test_data(self, seed, kshot, aus):
        inputa = []
        labela = []
        for au in aus:
            print('==== au: ', au)
            one_au_one_subject_on_frame_indices = self.on_info_df[au][0]
            random.seed(seed)
            selected_on_frame_idx = random.sample(one_au_one_subject_on_frame_indices,
                                                  min(kshot, len(one_au_one_subject_on_frame_indices)))
            print('-- selected_on_frame_idx: ', selected_on_frame_idx)

            one_au_one_subject_off_frame_indices = self.off_info_df[au][0]
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

            selected_idx = selected_off_frame_idx
            selected_idx.extend(selected_on_frame_idx)
            inputa.append(self.feat_vec[0][selected_idx])
            labela.append(self.labels[0][selected_idx])
        inputa = np.array(inputa)
        labela = np.array(labela)
        return inputa, inputa, labela, labela

    def sample_test_data_use_other_aus(self, seed, kshot, aus):
        inputa = []
        labela = []
        selected_frame_all = []
        # make frame idx set
        for au in aus:
            print('==== au: ', au)
            one_au_one_subject_on_frame_indices = self.on_info_df[au][0]
            random.seed(seed)
            selected_on_frame_idx = random.sample(one_au_one_subject_on_frame_indices,
                                                  min(kshot, len(one_au_one_subject_on_frame_indices)))
            print('-- selected_on_frame_idx: ', selected_on_frame_idx)
            selected_frame_all.extend(selected_on_frame_idx)

            one_au_one_subject_off_frame_indices = self.off_info_df[au][0]
            needed_num_samples = 2 * kshot - len(selected_on_frame_idx)
            random.seed(seed)
            selected_off_frame_idx = random.sample(one_au_one_subject_off_frame_indices, needed_num_samples)
            print('-- selected_off_frame_idx: ', selected_off_frame_idx)
            selected_frame_all.extend(selected_off_frame_idx)
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
        selected_frame_all = list(set(selected_frame_all))
        np.random.shuffle(selected_frame_all)

        for au in aus:
            inputa.append(self.feat_vec[0][selected_frame_all])
            labela.append(self.labels[0][selected_frame_all])
        inputa = np.array(inputa)
        labela = np.array(labela)
        return inputa, inputa, labela, labela, selected_frame_all

    def make_data_tensor(self, kshot_seed):
        print("===================================make_data_tensor in daga_generator2")
        print(">>>>>>> sampling seed: ", kshot_seed)
        folders = self.metatrain_character_folders
        print(">>>>>>> train folders: ", folders)

        # make list of files
        print('Generating filenames')
        # To have totally different inputa and inputb, they should be sampled at the same time and then splitted.
        for sub_folder in folders:  # 쓰일 task수만큼만 경로 만든다. 이 task들이 iteration동안 어차피 반복될거니까
            # random.shuffle(sampled_character_folders)
            off_imgs, on_imgs, off_labels, on_labels = test(sub_folder, FLAGS.feature_path,
                                                            kshot_seed,
                                                            nb_samples=FLAGS.update_batch_size)

        return inputa_latent_feat_tensor, inputb_latent_feat_tensor, labelas_tensor, labelbs_tensor
