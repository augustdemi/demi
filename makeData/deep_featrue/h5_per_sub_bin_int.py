import os
import cv2
import numpy as np
import h5py

path = "/home/ml1323/project/robert_data/DISFA/detected_disfa/"

for subject in os.listdir(path):
    print(">>>>>>>>> start: " + subject)
    img_arr = []
    img_files = os.listdir(path + subject)
    detected_frame_idx = [int(elt.split('frame')[1].split('_')[0]) for elt in img_files]
    detected_frame_idx = list(set(detected_frame_idx))
    for i in detected_frame_idx:
        img = cv2.imread(path + subject + "/frame" + str(i) + "_0.jpg")
        img_arr.append(img)

    ####### label ############
    label_path = "/home/ml1323/project/robert_data/DISFA/label/" + subject + "/"
    # label_path= "D:/연구/프로젝트/SN001/label/"

    all_au = ['au1', 'au2', 'au4', 'au6', 'au9', 'au12', 'au25', 'au26']

    label_for_all_au = []
    for au in all_au:  # file = intensity of each au
        intensities_for_one_au = []
        f = open(label_path + subject + '_' + au + '.txt', 'r')
        all_labels = f.readlines()
        for idx in detected_frame_idx:
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

    print('img_arr: ', len(img_arr))
    print('>>>before:', np.array(label_for_all_au).shape)
    final_label = np.array(label_for_all_au).transpose(1, 0, 2)
    print('>>>after:', final_label.shape)

    # min_len = np.min([len(detected_frame_idx), final_label.shape[0]])
    hf = h5py.File("/home/ml1323/project/robert_data/DISFA/h5_per_sub_bin_int/" + subject + ".h5", 'w')
    hf.create_dataset('img', data=img_arr)
    hf.create_dataset('lab', data=final_label)
    print("img :" + str(hf['img'].shape))
    print("lab :" + str(hf['lab'].shape))
    hf.close()
    print('done;', subject)
