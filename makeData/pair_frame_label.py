import os
import cv2
import numpy as np
import h5py


path = "/home/ml1323/project/robert_data/DISFA/detected_disfa/"

for subject in os.listdir(path):
    print(">>>>>>>>> start: " + subject)
    img_arr = []
    img_files=os.listdir(path + subject)
    detected_frame_idx = [int(elt.split('frame')[1].split('_')[0]) for elt in img_files]
    detected_frame_idx = list(set(detected_frame_idx))
    for i in detected_frame_idx:
        img = cv2.imread(path + subject + "/frame" + str(i) + "_0.jpg")
        img_arr.append(cv2.resize(img, (160, 240)))


    ####### label ############
    label_path= "/home/ml1323/project/robert_data/DISFA/label/" + subject + "/"
    #label_path= "D:/연구/프로젝트/SN001/label/"
    file_names = [subject + '_au1.txt', subject + '_au2.txt', subject + '_au4.txt', subject + '_au5.txt', subject + '_au6.txt', subject + '_au9.txt', subject + '_au12.txt', subject + '_au15.txt', subject + '_au17.txt', subject + '_au20.txt', subject + '_au25.txt', subject + '_au26.txt']

    label_for_all_au = []
    for file in file_names: # file = intensity of each au
        intensities_for_one_au = []
        with open(label_path + file, 'r') as f:
            for line in f:
                onehot_for_one_au = np.zeros(6) # intensity from 0 to 5
                intensity=int(line.split(",")[1].split("\n")[0])
                onehot_for_one_au[intensity] = 1
                intensities_for_one_au.append(onehot_for_one_au)
            label_for_all_au.append(intensities_for_one_au)
            f.close()

    final_label = np.array(label_for_all_au).transpose(1,0,2)

    min_len = np.min([len(detected_frame_idx), final_label.shape[0]])
    hf = h5py.File("/home/ml1323/project/robert_data/DISFA/h5_detected/"+subject+".h5", 'w')
    hf.create_dataset('img', data=img_arr[:min_len])
    hf.create_dataset('lab', data=final_label[detected_frame_idx[:min_len]])
    print("img :" + str(hf['img'].shape))
    print("lab :" + str(hf['lab'].shape))
    hf.close()

