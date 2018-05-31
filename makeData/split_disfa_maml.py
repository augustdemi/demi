import numpy as np
import os
import h5py
from sklearn.model_selection import KFold
import random


############### img ############
path = "/home/ml1323/project/robert_data/DISFA/h5_au12/"
# path = "D:/연구/프로젝트/DISFA/h5/"



files = os.listdir(path)
files.sort()
file_idx = np.array(range(0,len(files)))
train_index = file_idx[:14]
test_index = file_idx[14:]

print("TRAIN:", train_index, "TEST:", test_index)
data_idx = {'train': train_index, 'test': test_index}
for key in data_idx.keys():
    imgs = []
    labels =[]
    subjects =[]
    total_n = 0
    for i in data_idx[key]:
        file = files[i]
        print(">>>> file: " + file)
        hf = h5py.File(path+file, 'r')
        img=hf['img'].value
        label=hf['lab'].value
        hf.close()
        subject_number = int(file.split(".")[0].split("SN")[1])
        n_data = len(img)
        total_n += n_data
        print(img.shape)
        imgs.append(img)
        labels.append(label)
        subjects.append(subject_number * np.ones(n_data))
    save_path = "/home/ml1323/project/robert_data/DISFA/h5_maml/"
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    # save_path = "D:/연구/프로젝트/DISFA/rrr/"
    reshaped_imgs = imgs[0]
    reshaped_labels = labels[0]
    reshaped_subjects = subjects[0]
    for i in range(1, len(imgs)):
        reshaped_imgs = np.concatenate((reshaped_imgs , imgs[i]), axis=0)
        reshaped_labels = np.concatenate((reshaped_labels , labels[i]), axis=0)
        reshaped_subjects = np.concatenate((reshaped_subjects , subjects[i]), axis=0)
    random_idx = list(range(0,len(reshaped_imgs)))
    random.shuffle(random_idx)
    reshaped_imgs = reshaped_imgs[random_idx]
    reshaped_labels = reshaped_labels[random_idx]
    reshaped_subjects = reshaped_subjects[random_idx]

    hfs = h5py.File(save_path + key + ".h5", 'w')
    hfs.create_dataset('img', data=reshaped_imgs)
    hfs.create_dataset('lab', data=reshaped_labels)
    hfs.create_dataset('sub', data=reshaped_subjects)
    hfs.close()
    print(save_path + key + ".h5")





