import os
import cv2
import numpy as np
import h5py

# AU12에 대해서만 resized하지 않은 이미지와 intensity on/off적용한 라벨을 짝맞춰 h5파일 만듦 =>h5_au12에 저장
path = "/home/ml1323/project/robert_data/DISFA/detected_disfa/"
# path = "D:/연구/프로젝트/SN001/detected/"

for subject in os.listdir(path):
    subject="SN001"
    img_arr = []
    img_files=os.listdir(path + subject)
    detected_frame_idx = [int(elt.split('frame')[1].split('_')[0]) for elt in img_files]
    detected_frame_idx = list(set(detected_frame_idx))
    for i in detected_frame_idx:
        img = cv2.imread(path + subject + "/frame" + str(i) + "_0.jpg")
        img_arr.append(img)


    ####### label ############
    label_path= "/home/ml1323/project/robert_data/DISFA/label/" + subject + "/"
    # label_path= "D:/연구/프로젝트/SN001/label/"

    label_for_one_au = []
    with open(label_path + subject + '_au12.txt', 'r') as f:
        for line in f:
            intensity_onehot = [1, 0] # 0번째 index에 off(0-intensity), 1번째 index에 on(1-intensity)
            intensity = int(line.split(",")[1].split("\n")[0])
            if (intensity > 0):
                intensity_onehot = [0, 1]
            label_for_one_au.append(intensity_onehot)
        f.close()

    label_for_one_au = np.array(label_for_one_au)

    min_len = np.min([len(detected_frame_idx), label_for_one_au.shape[0]])
    hf = h5py.File("/home/ml1323/project/robert_data/DISFA/h5_binary_au12/"+subject+".h5", 'w')
    # hf = h5py.File(path+subject+".h5", 'w')
    hf.create_dataset('img', data=img_arr[:min_len])
    lab = label_for_one_au[detected_frame_idx[:min_len]]
    hf.create_dataset('lab', data=lab)
    print("img shape:", hf['img'].shape)
    print("lab shape:", hf['lab'].shape)
    on_indices = [j for j in range(len(lab)) if lab[j] == 1]
    print("AU12: " + str(i), len(on_indices))
    print(">>>>>>>>>>>>>>>>> end: " + subject)
    hf.close()

