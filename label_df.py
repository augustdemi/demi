import os
import numpy as np

label_folder = 'D:/연구/프로젝트/label'

subjects = os.listdir(label_folder)
subjects.sort()
subjects = subjects[0:5]
label_paths = [os.path.join(label_folder, subject) for subject in subjects]

print('>>>> get label')
aus = ['au1', 'au2', 'au4']
all_subject_labels = []  # list of feat_vec array per subject
all_subject_on_intensity_info = []
all_subject_off_intensity_info = []
for label_path in label_paths:
    subject = label_path.split('\\')[-1]
    print(subject)
    all_labels_per_subj = []
    all_au_on_intensity_info = []
    all_au_off_intensity_info = []
    for au in aus:
        f = open(os.path.join(label_path, subject + '_' + au + '.txt'), 'r')
        lines = f.readlines()[:4845]
        all_labels_per_subj.append([float(line.split(',')[1].split('\n')[0]) for line in lines])
        on_intensity_info = [i for i in range(len(lines)) if float(lines[i].split(',')[1].split('\n')[0]) > 0]
        off_intensity_info = [i for i in range(4845) if i not in on_intensity_info]
        all_au_on_intensity_info.append(on_intensity_info)
        all_au_off_intensity_info.append(off_intensity_info)
    all_labels_per_subj = np.transpose(np.array(all_labels_per_subj), (1, 0))
    all_subject_labels.append(all_labels_per_subj)
    all_subject_on_intensity_info.append(all_au_on_intensity_info)
    all_subject_off_intensity_info.append(all_au_off_intensity_info)
all_subject_on_intensity_info = np.array(all_subject_on_intensity_info)  # 14*8
all_subject_off_intensity_info = np.array(all_subject_off_intensity_info)

import pandas as pd

on_pd_data = {}
for i in range(len(aus)):
    on_pd_data[aus[i]] = all_subject_on_intensity_info[:, i]
on_df = pd.DataFrame(
    data=on_pd_data
)

off_pd_data = {}
for i in range(len(aus)):
    off_pd_data[aus[i]] = all_subject_off_intensity_info[:, i]
off_df = pd.DataFrame(
    data=off_pd_data
)

import random

kshot = 10
seed = 0

one_au_all_subjects_on_frame_indices = on_pd_data['au1']
selected_on_frame_idx = []
for each_subj_idx in one_au_all_subjects_on_frame_indices:
    random.seed(seed)
    selected_on_frame_idx.append(random.sample(each_subj_idx, min(2 * kshot, len(each_subj_idx))))

one_au_all_subjects_off_frame_indices = off_pd_data['au1']
selected_off_frame_idx = []
for i in range(len(one_au_all_subjects_off_frame_indices)):
    each_subj_idx = one_au_all_subjects_off_frame_indices[i]
    needed_num_samples = 4 * kshot - len(selected_on_frame_idx[i])
    selected_off_frame_idx.append(random.sample(each_subj_idx, needed_num_samples))

print(on_pd_data)
print(off_pd_data)
