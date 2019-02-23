import numpy as np
import os
import h5py
import random

# MAML을 위한 base model에서 쓰일 train/test 셋을 만들기위해 h5파일을 split함
############### img ############
path = "/home/ml1323/project/robert_data/DISFA/deep_feature200_dim2048/"
# path = "D:/연구/프로젝트/DISFA/h5/"
save_path = "/home/ml1323/project/robert_data/DISFA/h5_vae_feature_dim2048/"
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
    final_label = np.array(label_for_all_au).transpose(1, 0, 2)
    print('final_label: ', final_label.shape)
    return final_label


for key in data_idx.keys():
    features = []
    labels = []
    subjects = []
    for file in data_idx[key]:  # loop for each subject in TR or TE
        print(">>>> file: " + file)
        f = open(path + file, 'r')
        lines = f.readlines()
        all_features_per_sub = [np.array([float(elt) for elt in line.split(',')[2:]]) for line in
                                lines]  # 0 = subejct, 1=frame index
        frames = [line.split(',')[1] for line in lines]  # 0 = subejct, 1=frame index
        subject = lines[0].split(',')[0]
        if len(all_features_per_sub) == 4846: all_features_per_sub = all_features_per_sub[:4845]
        label = get_labet(subject, frames)
        f.close()
        subject_number = int(subject.split("SN")[1])
        n_data = len(all_features_per_sub)
        features.extend(all_features_per_sub)
        labels.extend(label)
        subjects.extend(subject_number * np.ones(n_data))

    random_idx = list(range(0, len(features)))
    random.shuffle(random_idx)
    features = np.array(features)[random_idx]
    labels = np.array(labels)[random_idx]
    subjects = np.array(subjects)[random_idx]
    print('shape of feat: ', features.shape)
    print('shape of labels: ', labels.shape)
    print('shape of subjects: ', subjects.shape)
    hfs = h5py.File(save_path + key + ".h5", 'w')
    hfs.create_dataset('feat', data=features)
    hfs.create_dataset('lab', data=labels)
    hfs.create_dataset('sub', data=subjects)
    hfs.close()
    print(save_path + key + ".h5")
    print("=========================================")
