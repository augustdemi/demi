import numpy as np
import os
import h5py
import random

# MAML을 위한 base model에서 쓰일 train/test 셋을 만들기위해 h5파일을 split함
############### img ############
path = "/home/ml1323/project/robert_data/DISFA/h5_per_sub_bin_int/"
# path = "D:/연구/프로젝트/DISFA/h5/"



files = []

for file_name in os.listdir(path):
    if (file_name.startswith("outa")):
        files.append((int(file_name.split(".")[0].split("_")[1]), file_name))

files.sort(key=lambda f: f[0])

print("TRAIN:", files[:14], "TEST:", files[14:])
data_idx = {'train': files[:14], 'test': files[14:]}
for key in data_idx.keys():
    imgs = []
    labels = []
    subjects = []
    total_n = 0
    for file in files:
        file = files[i]
        print(">>>> file: " + file)
        hf = h5py.File(path + file, 'r')
        img = hf['img'].value
        if len(img) == 4846: img = img[:4845]
        label = hf['lab'].value
        hf.close()
        subject_number = int(file.split(".")[0].split("SN")[1])
        n_data = len(img)
        total_n += n_data
        print(img.shape)
        imgs.append(img)
        labels.append(label)
        subjects.append(subject_number * np.ones(n_data))
    save_path = "/home/ml1323/project/robert_data/DISFA/h5_vae_img/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save_path = "D:/연구/프로젝트/DISFA/rrr/"
    reshaped_imgs = imgs[0]
    reshaped_labels = labels[0]
    reshaped_subjects = subjects[0]
    for i in range(1, len(imgs)):
        reshaped_imgs = np.concatenate((reshaped_imgs, imgs[i]), axis=0)
        reshaped_labels = np.concatenate((reshaped_labels, labels[i]), axis=0)
        reshaped_subjects = np.concatenate((reshaped_subjects, subjects[i]), axis=0)
    random_idx = list(range(0, len(reshaped_imgs)))
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
    print("=========================================")
