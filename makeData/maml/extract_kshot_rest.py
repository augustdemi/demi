import h5py as h
import random
import os
from shutil import copyfile
import numpy as np
#on/off intensity를 라벨로 매칭시킨 h5로부터 maml을 위한 이미지 경로와 이미지 파일 생성

# original_frame_path = "D:/연구/프로젝트/SN001/frames/"
original_frame_path = "/home/ml1323/project/robert_data/DISFA/detected_disfa/"
k_shot = 50
for subject in os.listdir(original_frame_path):
    # subject = "SN001"
    ####### label ############
    # label_path= "D:/연구/프로젝트/DISFA/label/" + subject + "/" + subject + "_au12.txt"
    label_path = "/home/ml1323/project/robert_data/DISFA/label/" + subject + "/" + subject + "_au12.txt"
    intensities_for_one_au = []
    with open(label_path, 'r') as f:
        for line in f:
            intensity = int(line.split(",")[1].split("\n")[0])
            if intensity > 0: intensity = 1
            intensities_for_one_au.append(intensity)
        f.close()

    lab_au12 = np.array(intensities_for_one_au)

    detected_img_files = os.listdir(original_frame_path + subject)
    detected_frame_idx = [int(elt.split('frame')[1].split('_')[0]) for elt in detected_img_files]
    detected_frame_idx = list(set(detected_frame_idx))
    min_len = np.min([len(detected_frame_idx), len(lab_au12)])

    total_on_intst_idx = [i for i in detected_frame_idx[:min_len] if lab_au12[i] == 1]
    print(total_on_intst_idx)
    print(">>>>>>>>>>>", len(total_on_intst_idx))
    total_off_intst_idx = [i for i in detected_frame_idx[:min_len] if lab_au12[i] == 0]

    random_on_idx = random.sample(total_on_intst_idx, 2 * k_shot)
    random_off_idx = random.sample(total_off_intst_idx, 2 * k_shot)

    rest_on_idx = [i for i in total_on_intst_idx if i not in random_on_idx]
    rest_off_idx = [i for i in total_off_intst_idx if i not in random_off_idx]


    kshot_path = "/home/ml1323/project/robert_data/DISFA/kshot_rest/"
    if not os.path.exists(kshot_path + "kshot/" + subject + "/on"): os.makedirs(kshot_path + "kshot/" + subject + "/on")
    if not os.path.exists(kshot_path + "kshot/" + subject + "/off"): os.makedirs(kshot_path + "kshot/" + subject + "/off")

    # copy on intensity frames to kshot folder
    for i in random_on_idx:
        copyfile(original_frame_path + subject + "/frame" + str(i) + "_0.jpg",
                 kshot_path + "kshot/" + subject + "/on/frame" + str(i) + ".jpg")

    # copy off intensity frames to kshot folder
    for i in random_off_idx:
        copyfile(original_frame_path + subject + "/frame" + str(i) + "_0.jpg",
                 kshot_path + "kshot/" + subject + "/off/frame" + str(i) + ".jpg")

    if not os.path.exists(kshot_path + "rest/" + subject + "/on"): os.makedirs(kshot_path + "rest/" + subject + "/on")
    if not os.path.exists(kshot_path + "rest/" + subject + "/off"): os.makedirs(kshot_path + "rest/" + subject + "/off")

    # copy on intensity frames to kshot folder
    for i in rest_on_idx:
        copyfile(original_frame_path + subject + "/frame" + str(i) + "_0.jpg",
                 kshot_path + "rest/" + subject + "/on/frame" + str(i) + ".jpg")

    # copy off intensity frames to kshot folder
    for i in rest_off_idx:
        copyfile(original_frame_path + subject + "/frame" + str(i) + "_0.jpg",
                 kshot_path + "rest/" + subject + "/off/frame" + str(i) + ".jpg")