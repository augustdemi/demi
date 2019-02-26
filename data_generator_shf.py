""" Code for loading data. """
import numpy as np
import os

from tensorflow.python.platform import flags
from utils import get_all_feature_w_all_labels
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
        subject_index = [int(elt) for elt in FLAGS.subject_index.split(',')]
        print('>>>>>>>>>>>>>> subject index: ', subject_index)
        subjects = np.array(subjects)[subject_index]
        print('>>>>>>>>>>>>>> selected subjects from feat_vec: ', subjects)
        self.feature_files = [os.path.join(data_folder, subject) for subject in subjects]

        label_folder = FLAGS.labeldir  # label_folder = '/home/ml1323/project/robert_data/DISFA/label/'
        subjects = os.listdir(label_folder)
        subjects.sort()
        subjects = np.array(subjects)[subject_index]
        self.label_folder = [os.path.join(label_folder, subject) for subject in subjects]

        feat_vec, labels, on_info_df, off_info_df, test_b_frame = get_all_feature_w_all_labels(self.feature_files,
                                                                                               self.label_folder,
                                                                                               FLAGS.test_split_seed)

        self.feat_vec = feat_vec
        self.labels = labels
        self.on_info_df = on_info_df
        self.off_info_df = off_info_df
        self.test_b_frame = test_b_frame
        print('========== will be used this test_b ===========')
        print(len(test_b_frame))
        print(test_b_frame)

    def shuffle_data(self, seed, kshot, aus):
        print('>>>>>> sampling way: inputa != inputb')
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
            # print('>>> selected_on_frame_idx: ', selected_on_frame_idx)

            one_au_all_subjects_off_frame_indices = self.off_info_df[au]
            selected_off_frame_idx = []
            for i in range(len(one_au_all_subjects_off_frame_indices)):
                each_subj_idx = one_au_all_subjects_off_frame_indices[i]
                needed_num_samples = 4 * kshot - len(selected_on_frame_idx[i])
                random.seed(seed)
                selected_off_frame_idx.append(random.sample(each_subj_idx, needed_num_samples))
            # print('>>> selected_off_frame_idx: ', selected_off_frame_idx)


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


    def sample_test_data_use_all(self, seed, kshot, aus):
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

        selected_frame_all = list(set(selected_frame_all))
        np.random.shuffle(selected_frame_all)
        if FLAGS.check_sample:
            save_path = '/home/ml1323/project/robert_code/new/check_labels/test/' + str(
                FLAGS.update_batch_size) + 'shot'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        import csv
        with open(save_path + '/subject' + FLAGS.subject_index + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(selected_frame_all.join(','))

        for _ in aus:
            inputa.append(self.feat_vec[0][selected_frame_all])
            labela.append(self.labels[0][selected_frame_all])
        inputa = np.array(inputa)
        labela = np.array(labela)
        return inputa, inputa, labela, labela, selected_frame_all
