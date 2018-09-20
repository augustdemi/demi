import os
import numpy as np

path = "/home/ml1323/project/robert_data/DISFA/new_dataset/test_b/"
save_path = path + "/testset/"
if not os.path.exists(save_path): os.makedirs(save_path)

subjects = os.listdir(path)
subjects.sort()
all_au = ['au1', 'au2', 'au4', 'au5', 'au6', 'au9', 'au12', 'au15', 'au17', 'au20', 'au25', 'au26']

import cv2

total_img = []
total_subj = []
total_label = []

for subject in subjects:
    subject_folder = os.path.join(path, subject)
    files = os.listdir(subject_folder)
    test_file_names = [os.path.join(subject_folder, file) for file in files]
    print('original files len in ', subject, ' : ', len(files))
    img_arr = []
    for filename in test_file_names:
        img = cv2.imread(filename)
        img_arr.append(img)
    frame_idx = [int(file.split('frame')[1].split('.')[0]) for file in files]
    total_img.extend(img_arr)

    label_per_subject = []
    for au in all_au:
        label_path = "/home/ml1323/project/robert_data/DISFA/label/" + subject + "/" + subject + "_" + au + ".txt"
        intensities_for_one_au = []
        f = open(label_path, 'r')
        all_labels = f.readlines()

        label_per_au = []
        for idx in frame_idx:
            try:
                intensity = int(all_labels[idx].split(",")[1].split("\n")[0])
            except:
                print("on_idx - out of index: ", idx, " for au, subject: ", au, subject)
                continue
            if intensity > 0:
                label_per_au.append([0, 1])
            else:
                label_per_au.append([1, 0])
        # label_per_au = (4800,2)
        label_per_subject.append(label_per_au)  # (12, 4800,2)
    label_per_subject = np.array(label_per_subject)
    print('label_per_subject shape: ', label_per_subject.shape)
    print(label_per_subject[:3])

    label_per_subject = np.transpose(label_per_subject, (1, 0, 2))  # (4800,12,2)
    print('label_per_subject shape after : ', label_per_subject.shape)
    total_label.extend(label_per_subject)
    subject_number = int(subject.split(".")[0].split("SN")[1])
    print("subject_number: ", subject_number)
    total_subj.extend(subject_number * np.ones(len(frame_idx)))

    print(label_per_subject[:3])
    print('>>>>>>>>>>>>>>>>>>>>>>>> sub done:', subject)

import h5py

hfs = h5py.File(save_path + "testtest.h5", 'w')
np.random.seed(3)
np.random.shuffle(total_img)
np.random.seed(3)
np.random.shuffle(total_label)
np.random.seed(3)
np.random.shuffle(total_subj)

hfs.create_dataset('img', data=total_img)
hfs.create_dataset('lab', data=total_label)
hfs.create_dataset('sub', data=total_subj)
hfs.close()
