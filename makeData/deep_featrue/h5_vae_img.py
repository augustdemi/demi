import numpy as np
import os
import h5py
import random

# MAML을 위한 base model에서 쓰일 train/test 셋을 만들기위해 h5파일을 split함
############### img ############
path = "/home/ml1323/project/robert_data/DISFA_new/h5_per_sub_bin_int/"
# path = "D:/연구/프로젝트/DISFA/h5/"

save_path = "/home/ml1323/project/robert_data/DISFA_new/h5_vae_img/fold1_val2/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
val_ratio = 0.3

files = []

for file_name in os.listdir(path):
    files.append((int(file_name.split(".")[0].split("SN0")[1]), file_name))


files.sort(key=lambda f: f[0])

train_set = files[:2]
test_set = files[18:]

data_idx = {'train': [f[1] for f in train_set], 'test': [f[1] for f in test_set]}
print(data_idx)

# imgs = []
# labels = []
# subjects = []
# frames = []
# for file in data_idx['test']:
#     print(">>>> file: " + file)
#     hf = h5py.File(path + file, 'r')
#     img = hf['img'].value
#     frame = hf['frame'].value
#     if len(img) == 4846:
#         img = img[:4845]
#         frame = frame[:4845]
#     label = hf['lab'].value
#     hf.close()
#     subject_number = int(file.split(".")[0].split("SN")[1])
#     n_data = len(img)
#     print(img.shape)
#     imgs.append(img)
#     labels.append(label)
#     frames.append(frame)
#     subjects.append(subject_number * np.ones(n_data))
#
# reshaped_imgs = imgs[0]
# reshaped_labels = labels[0]
# reshaped_subjects = subjects[0]
# reshaped_frames = frames[0]
# for i in range(1, len(imgs)):
#     reshaped_imgs = np.concatenate((reshaped_imgs, imgs[i]), axis=0)
#     reshaped_labels = np.concatenate((reshaped_labels, labels[i]), axis=0)
#     reshaped_subjects = np.concatenate((reshaped_subjects, subjects[i]), axis=0)
#     reshaped_frames = np.concatenate((reshaped_frames, frames[i]), axis=0)
# random_idx = list(range(0, len(reshaped_imgs)))
# random.shuffle(random_idx)
# reshaped_imgs = reshaped_imgs[random_idx]
# reshaped_labels = reshaped_labels[random_idx]
# reshaped_subjects = reshaped_subjects[random_idx]
# reshaped_frames = reshaped_frames[random_idx]
#
# hfs = h5py.File(save_path + "test.h5", 'w')
# hfs.create_dataset('img', data=reshaped_imgs)
# hfs.create_dataset('lab', data=reshaped_labels)
# hfs.create_dataset('sub', data=reshaped_subjects)
# hfs.create_dataset('frame', data=reshaped_frames)
# hfs.close()
# print(save_path + "test.h5")
# print("=========================================")
#
#



train_imgs = []
train_labels = []
train_subjects = []
train_frames = []
train_frame_info = {}

val_imgs = []
val_labels = []
val_subjects = []
val_frames = []
val_frame_info = {}
for file in data_idx['train']:
    print(">>>> file: " + file)
    hf = h5py.File(path + file, 'r')
    one_sub_img = hf['img'].value
    one_sub_frame = hf['frame'].value
    one_sub_lab = hf['lab'].value
    hf.close()
    if len(one_sub_img) == 4846:
        one_sub_img = one_sub_img[:4845]
        one_sub_frame = one_sub_frame[:4845]
    subject_number = int(file.split(".")[0].split("SN")[1])
    total_num_data_one_sub = len(one_sub_img)
    total_idx = list(range(total_num_data_one_sub))
    random.seed(0)
    val_idx = random.sample(total_idx, int(total_num_data_one_sub * val_ratio))
    train_idx = [i for i in total_idx if i not in val_idx]
    #add train-val
    val_imgs.extend(one_sub_img[val_idx])
    val_labels.extend(one_sub_lab[val_idx])
    val_frames.extend(one_sub_frame[val_idx])
    val_subjects.extend(np.array([subject_number]*len(val_idx)))
    val_frame_info[subject_number] = list(one_sub_frame[val_idx])
    print(list(one_sub_frame[val_idx]))

    #add train-train
    train_imgs.extend(one_sub_img[train_idx])
    train_labels.extend(one_sub_lab[train_idx])
    train_frames.extend(one_sub_frame[train_idx])
    train_subjects.extend(np.array([subject_number]*len(train_idx)))
    train_frame_info[subject_number] = list(one_sub_frame[train_idx])

print('///////////////')
print(val_frame_info)
print('///////////////')

random_idx = list(range(len(val_imgs)))
random.seed(0)
random.shuffle(random_idx)
val_imgs = np.array(val_imgs)[random_idx]
val_labels = np.array(val_labels)[random_idx]
val_frames = np.array(val_frames)[random_idx]
val_subjects = np.array(val_subjects)[random_idx]

hfs = h5py.File(save_path + "train-val.h5", 'w')
hfs.create_dataset('img', data=val_imgs)
hfs.create_dataset('lab', data=val_labels)
hfs.create_dataset('frame', data=val_frames)
hfs.create_dataset('sub', data=val_subjects)
hfs.close()
print("=========================================")

# random_idx = list(range(len(train_imgs)))
# random.seed(0)
# random.shuffle(random_idx)
# train_imgs = np.array(train_imgs)[random_idx]
# train_labels = np.array(train_labels)[random_idx]
# train_frames = np.array(train_frames)[random_idx]
# train_subjects = np.array(train_subjects)[random_idx]
#
# hfs = h5py.File(save_path + "train-train.h5", 'w')
# hfs.create_dataset('img', data=train_imgs)
# hfs.create_dataset('lab', data=train_labels)
# hfs.create_dataset('frame', data=train_frames)
# hfs.create_dataset('sub', data=train_subjects)
# hfs.close()

import json
with open(save_path + 'val_frame_info.json', 'w') as outfile:
    json.dump(val_frame_info, outfile)
with open(save_path + 'train_frame_info.json', 'w') as outfile:
    json.dump(train_frame_info, outfile)