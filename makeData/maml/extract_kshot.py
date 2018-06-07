import h5py as h
import random
import os
from shutil import copyfile
import numpy as np
#on/off intensity를 라벨로 매칭시킨 h5로부터 maml을 위한 이미지 경로와 이미지 파일 생성

# original_frame_path = "D:/연구/프로젝트/SN001/frames/"
original_frame_path = "/home/ml1323/project/robert_data/DISFA/detected_disfa/"
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

    on_intst_idx = [i for i in detected_frame_idx[:min_len] if lab_au12[i] == 1]
    print(on_intst_idx)
    off_intst_idx = [i for i in detected_frame_idx[:min_len] if lab_au12[i] == 0]

    n_class = 2
    k_shot = 10
    k_fold = 3

    adap_on_pool = on_intst_idx[:int(len(on_intst_idx) / 2)]
    meta_on_pool = on_intst_idx[int(len(on_intst_idx) / 2):]
    adap_off_pool = off_intst_idx[:int(len(off_intst_idx) / 2)]
    meta_off_pool = off_intst_idx[int(len(off_intst_idx) / 2):]

    adap_on_idx = random.sample(adap_on_pool, n_class * k_shot * k_fold)
    meta_on_idx = random.sample(meta_on_pool, n_class * k_shot * k_fold)
    adap_off_idx = random.sample(adap_off_pool, n_class * k_shot * k_fold)
    meta_off_idx = random.sample(meta_off_pool, n_class * k_shot * k_fold)

    on_idx = np.append(adap_on_idx, meta_on_idx)
    off_idx = np.append(adap_off_idx, meta_off_idx)

    for k in range(k_fold):
        # kshot_path = "D:/연구/프로젝트/SN001/maml/kshot/" + str(k)
        kshot_path = "/home/ml1323/project/robert_data/DISFA/kshot/" + str(k) + "/" + subject
        if not os.path.exists(kshot_path + "/on"): os.makedirs(kshot_path + "/on")
        if not os.path.exists(kshot_path + "/off"): os.makedirs(kshot_path + "/off")

        # copy on intensity frames to kshot folder
        for i in on_idx[k * n_class * k_shot: (k + 1) * n_class * k_shot]:
            copyfile(original_frame_path + subject + "/frame" + str(i) + "_0.jpg",
                     kshot_path + "/on/frame" + str(i) + ".jpg")

        # copy off intensity frames to kshot folder
        for i in off_idx[k * n_class * k_shot: (k + 1) * n_class * k_shot]:
            copyfile(original_frame_path + subject + "/frame" + str(i) + "_0.jpg",
                     kshot_path + "/off/frame" + str(i) + ".jpg")
