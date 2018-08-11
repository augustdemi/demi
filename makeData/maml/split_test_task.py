import h5py as h
import random
import os
from shutil import copyfile
import numpy as np

# on/off intensity를 라벨로 매칭시킨 h5로부터 maml을 위한 이미지 경로와 이미지 파일 생성

# original_frame_path = "D:/연구/프로젝트/SN001/frames/"
original_frame_path = "/home/ml1323/project/robert_data/DISFA/detected_disfa/"
all_au = ['au1', 'au2', 'au4', 'au5', 'au6', 'au9', 'au12', 'au15', 'au17', 'au20', 'au25', 'au26']
test_subjects = ['SN017', 'SN018', 'SN021', 'SN023', 'SN024', 'SN025', 'SN026', 'SN027', 'SN028', 'SN029', 'SN030',
                 'SN031', 'SN032']

for subject in test_subjects:
    detected_img_files = os.listdir(original_frame_path + subject)
    detected_frame_idx = [int(elt.split('frame')[1].split('_')[0]) for elt in detected_img_files]
    detected_frame_idx = list(set(detected_frame_idx))
    # 홀수개면 한개 빼버림
    if type(len(detected_frame_idx) / 2) is not int: detected_frame_idx = detected_frame_idx[:-1]
    min_len = len(detected_frame_idx)
    print('min_len: ', min_len)
    test_a_idx = random.sample(detected_frame_idx, int(min_len / 2))
    test_b_idx = [i for i in detected_frame_idx if i not in test_a_idx]
    print('test_a len: ', len(test_a_idx))
    print('test_b len: ', len(test_b_idx))
    print('-------------------------------------------------')
    print(test_a_idx)
    print('-------------------------------------------------')
    print(test_b_idx)
    ########### 위의 각 subject 별 fixed test_a, test_b에 대해서, 각 au별로 on/off를 구해 분리해둠.
    ########### 그리곤 test_b셋에대해서 testset pickle값을 만들어 둬서 이 전체 데이터 셋에 대해서 evaluation할 것임.

    for au in all_au:
        label_path = "/home/ml1323/project/robert_data/DISFA/label/" + subject + "/" + subject + "_" + au + ".txt"
        intensities_for_one_au = []
        f = open(label_path, 'r')
        all_labels = f.readlines()

        test_a_on_idx = []
        test_a_off_idx = []
        for idx in test_a_idx:
            intensity = int(all_labels[idx].split(",")[1].split("\n")[0])
            if intensity > 0:
                test_a_on_idx.append(idx)
            else:
                test_a_off_idx.append(idx)

        test_b_on_idx = []
        test_b_off_idx = []
        for idx in test_b_idx:
            intensity = int(all_labels[idx].split(",")[1].split("\n")[0])
            if intensity > 0:
                test_b_on_idx.append(idx)
            else:
                test_b_off_idx.append(idx)
        f.close()

        print('test_a_on_idx len: ', len(test_a_on_idx))
        print('test_a_off_idx len: ', len(test_a_off_idx))
        print('test_b_on_idx len: ', len(test_b_on_idx))
        print('test_b_off_idx len: ', len(test_b_off_idx))

        save_path = "/home/ml1323/project/robert_data/DISFA/new_dataset/"
        if not os.path.exists(save_path + "/test_a/" + au + " / " + subject + "/on"): os.makedirs(
            save_path + "/test_a/" + au + " / " + subject + "/on")
        if not os.path.exists(save_path + "/test_a/" + au + " / " + subject + "/off"): os.makedirs(
            save_path + "/test_a/" + au + " / " + subject + "/off")
        if not os.path.exists(save_path + "/test_b/" + au + " / " + subject + "/on"): os.makedirs(
            save_path + "/test_b/" + au + " / " + subject + "/on")
        if not os.path.exists(save_path + "/test_b/" + au + " / " + subject + "/off"): os.makedirs(
            save_path + "/test_b/" + au + " / " + subject + "/off")

        # copy on intensity frames for test_a
        for i in test_a_on_idx:
            copyfile(original_frame_path + subject + "/frame" + str(i) + "_0.jpg",
                     save_path + "/test_a/" + au + " / " + subject + "/on/frame" + str(i) + ".jpg")

        # copy off intensity frames for test_a
        for i in test_a_off_idx:
            copyfile(original_frame_path + subject + "/frame" + str(i) + "_0.jpg",
                     save_path + "/test_a/" + au + " / " + subject + "/off/frame" + str(i) + ".jpg")

        # copy on intensity frames for test_b
        for i in test_b_on_idx:
            copyfile(original_frame_path + subject + "/frame" + str(i) + "_0.jpg",
                     save_path + "/test_b/" + au + " / " + subject + "/on/frame" + str(i) + ".jpg")

        # copy off intensity frames for test_b
        for i in test_b_off_idx:
            copyfile(original_frame_path + subject + "/frame" + str(i) + "_0.jpg",
                     save_path + "/test_b/" + au + " / " + subject + "/off/frame" + str(i) + ".jpg")
        print(">>>>> done: ", au)
