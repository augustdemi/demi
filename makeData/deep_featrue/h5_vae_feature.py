import numpy as np
import os
import h5py
import random

# MAML을 위한 base model에서 쓰일 train/test 셋을 만들기위해 h5파일을 split함
############### img ############
path = "/home/ml1323/project/robert_data/DISFA/deep_feature200/"
# path = "D:/연구/프로젝트/DISFA/h5/"
save_path = "/home/ml1323/project/robert_data/DISFA/h5_vae_feature/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

files = []

for file_name in os.listdir(path):
    files.append((int(file_name.split(".")[0].split("SN0")[1]), file_name))

files.sort(key=lambda f: f[0])
data_idx = {'train': [f[1] for f in files[:14]], 'test': [f[1] for f in files[14:]]}
print(data_idx)


def get_labet(subject, frames):
    label_path = "/home/ml1323/project/robert_data/DISFA/label/" + subject + "/"
    # label_path= "D:/연구/프로젝트/SN001/label/"
    all_au = ['au1', 'au2', 'au4', 'au6', 'au9', 'au12', 'au25', 'au26']

    frame_idx = [int(frame.split('frame')[1]) for frame in frames]
    label_for_all_au = []
    for au in all_au:  # file = intensity of each au
        intensities_for_one_au = []
        f = open(label_path + subject + '_' + au + '.txt', 'r')
        all_labels = f.readlines()
        for idx in frame_idx:
            intensity_onehot = np.zeros(2)
            try:
                intensity = int(all_labels[idx].split(",")[1].split("\n")[0])
            except:
                print("on_idx - out of index: ", idx, " for subject: ", subject)
                continue
            if (intensity > 0):
                intensity_onehot = [0, 1]
            else:
                intensity_onehot = [1, 0]
            intensities_for_one_au.append(intensity_onehot)

        label_for_all_au.append(intensities_for_one_au)
        f.close()
        print('done;', au)
    return label_for_all_au


for key in data_idx.keys():
    features = []
    labels = []
    subjects = []
    for file in data_idx[key]:  # loop for each subject in TR or TE
        print(">>>> file: " + file)
        f = open(path + file, 'r')
        lines = f.readlines()
        all_features_per_sub = [line.split(',')[2:] for line in lines]  # 0 = subejct, 1=frame index
        frames = [line.split(',')[1] for line in lines]  # 0 = subejct, 1=frame index
        subject = lines[0].split(',')[0]
        if len(all_features_per_sub) == 4846: all_features_per_sub = all_features_per_sub[:4845]
        label = get_labet(subject, frames)
        f.close()
        subject_number = int(subject.split("SN")[1])
        n_data = len(all_features_per_sub)
        features.append(all_features_per_sub)
        labels.append(label)
        subjects.append(subject_number * np.ones(n_data))

    # save_path = "D:/연구/프로젝트/DISFA/rrr/"
    reshaped_features = features[0]
    reshaped_labels = labels[0]
    reshaped_subjects = subjects[0]
    for i in range(1, len(features)):
        reshaped_features = np.concatenate((reshaped_features, features[i]), axis=0)
        reshaped_labels = np.concatenate((reshaped_labels, labels[i]), axis=0)
        reshaped_subjects = np.concatenate((reshaped_subjects, subjects[i]), axis=0)
    random_idx = list(range(0, len(reshaped_features)))
    random.shuffle(random_idx)
    reshaped_features = reshaped_features[random_idx]
    reshaped_labels = reshaped_labels[random_idx]
    reshaped_subjects = reshaped_subjects[random_idx]
    print('reshaped_features', reshaped_features.shape)
    print('reshaped_labels', reshaped_labels.shape)
    print('reshaped_subjects', reshaped_subjects.shape)

    hfs = h5py.File(save_path + key + ".h5", 'w')
    hfs.create_dataset('feat', data=reshaped_features)
    hfs.create_dataset('lab', data=reshaped_labels)
    hfs.create_dataset('sub', data=reshaped_subjects)
    hfs.close()
    print(save_path + key + ".h5")
    print("=========================================")
