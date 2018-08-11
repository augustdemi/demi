import h5py as h
import random
import os
from shutil import copyfile
import numpy as np

# on/off intensity를 라벨로 매칭시킨 h5로부터 maml을 위한 이미지 경로와 이미지 파일 생성

# original_frame_path = "D:/연구/프로젝트/SN001/frames/"
original_frame_path = "/home/ml1323/project/robert_data/DISFA/detected_disfa/"
all_au = ['au1', 'au2', 'au4', 'au5', 'au6', 'au9', 'au12', 'au15', 'au17', 'au20', 'au25', 'au26']
train_subjects = ['SN001', 'SN002', 'SN003', 'SN004', 'SN005', 'SN006', 'SN007', 'SN008', 'SN009', 'SN010', 'SN011',
                 'SN012', 'SN013', 'SN016']
for au in all_au:
    for subject in train_subjects:
        detected_img_files = os.listdir(original_frame_path + subject)
        detected_frame_idx = [int(elt.split('frame')[1].split('_')[0]) for elt in detected_img_files]
        detected_frame_idx = list(set(detected_frame_idx))

        label_path = "/home/ml1323/project/robert_data/DISFA/label/" + subject + "/" + subject + "_" + au + ".txt"
        intensities_for_one_au = []
        f = open(label_path, 'r')
        all_labels = f.readlines()

        train_on_idx = []
        train_off_idx = []
        for idx in detected_frame_idx:
            intensity = int(all_labels[idx].split(",")[1].split("\n")[0])
            if intensity > 0:
                train_on_idx.append(idx)
            else:
                train_off_idx.append(idx)

        print('train_on_idx len: ', len(train_on_idx))
        print('train_off_idx len: ', len(train_off_idx))

        save_path = "/home/ml1323/project/robert_data/DISFA/new_dataset/train/" + au + "/" + subject
        if not os.path.exists(save_path + "/on"): os.makedirs(save_path + "/on")
        if not os.path.exists(save_path + "/off"): os.makedirs(save_path + "/off")

        # copy on intensity frames for train
        for i in train_on_idx:
            copyfile(original_frame_path + subject + "/frame" + str(i) + "_0.jpg",
                     save_path + "/on/frame" + str(i) + ".jpg")

        # copy off intensity frames for train
        for i in train_off_idx:
            copyfile(original_frame_path + subject + "/frame" + str(i) + "_0.jpg",
                     save_path + "/off/frame" + str(i) + ".jpg")
        print(">>>>> done: ", au)
