import cv2
import os
import numpy as np
import h5py
# image_path = 'frame0.jpg'
# img = cv2.imread(image_path, 1)
# print (os.path.exists(image_path))
# print(img.shape)
# img = cv2.resize(img,(160,240))
#


############### img ############
video_path = "/bigdata1/homedirs/ml1323/project/robert_data/DISFA/video/"
# subject_list = []
# for i in range(0,27):
#     subject_list.append("SN0" + str(i).zfill(2))
#path = "D:/연구/프로젝트/SN001/RightVideoSN001_Comp.avi"


#files = os.listdir(video_path)
files = ["RightVideoSN001_comp.avi", "RightVideoSN002_comp.avi"]
all_img = []
all_label = []
for file in files:
    subject = "SN" + file.split("_")[0].split("SN")[1]
    print(">>>>>>>>> start: " + subject)
    vidcap = cv2.VideoCapture(video_path + "RightVideo" + subject + "_comp.avi")
    success, image = vidcap.read()
    count = 0
    success = True
    img_arr = []
    while success:
        resized_img = cv2.resize(image, (160, 240))
        img_arr.append(resized_img)
        success, image = vidcap.read()
        count += 1
    print(count)
    img_arr = np.array(img_arr)


    ####### label ############
    label_path= "/bigdata1/homedirs/ml1323/project/robert_data/DISFA/label/" + subject + "/"
    files = os.listdir(label_path)
    file_names = [subject + '_au1.txt', subject + '_au2.txt', subject + '_au4.txt', subject + '_au5.txt', subject + '_au6.txt', subject + '_au9.txt', subject + '_au12.txt', subject + '_au15.txt', subject + '_au17.txt', subject + '_au20.txt', subject + '_au25.txt', subject + '_au26.txt']

    label_for_all_au = []
    for file in file_names:
        label_for_one_au = []
        with open(label_path + file, 'r') as f:
            for line in f:
                label=int(line.split(",")[1].split("\n")[0])
                onehot_label = np.zeros(6)
                onehot_label[label] = 1
                label_for_one_au.append(onehot_label)
            label_for_all_au.append(label_for_one_au)
            f.close()

    final_label = np.array(label_for_all_au).transpose(1,0,2)
    all_img.append(img_arr)
    all_label.append(final_label)



hf = h5py.File("/bigdata1/homedirs/ml1323/project/robert_data/DISFA/total.h5", 'w')
hf.create_dataset('img', data=np.vstack(all_img))
hf.create_dataset('lab', data=np.vstack(all_label))
hf.close()


