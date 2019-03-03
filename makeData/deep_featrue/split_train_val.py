import numpy as np
import os
import h5py
import random

val_ratio = 0.3
file_path = "/home/ml1323/project/robert_data/DISFA_new/h5_vae_img/fold1/train.h5"
save_path = "/home/ml1323/project/robert_data/DISFA_new/h5_vae_img/fold1/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

files = []
hf = h5py.File(file_path, 'r')

total_num_data = hf['img'].shape[0]
all_subjects = list(set(hf['sub']))

total_val_img = []
total_val_label = []
total_val_sub = []
total_train_img = []
total_train_label = []
total_train_sub = []
for subject in all_subjects:
    # one_sub_info = [(hf['img'][i], hf['lab'][i], s_idx) for i in range(total_num_data) if hf['sub'][i] == s_idx]

    img = []
    lab = []
    for i in range(total_num_data):
        if hf['sub'][i] == subject:
            img.append(hf['img'][i])
            lab.append(hf['lab'][i])
    total_num_data_one_sub = len(img)
    total_idx = list(range(total_num_data_one_sub))
    val_idx = random.sample(total_idx, int(total_num_data_one_sub * val_ratio))
    train_idx = [i for i in total_idx if i not in val_idx]
    print('-----------------------------------------------------------------------------')
    print('num of validation set of subject {} is {}'.format(subject, len(val_idx)))
    print('-----------------------------------------------------------------------------')

    one_sub_img = np.array(img)
    one_sub_lab = np.array(lab)
    #add train-val
    total_val_img.append(one_sub_img[val_idx])
    total_val_label.append(one_sub_lab[val_idx])
    total_val_sub.append([subject]*len(val_idx))
    #add train-train
    total_train_img.append(one_sub_img[train_idx])
    total_train_label.append(one_sub_lab[train_idx])
    total_train_sub.append([subject] * len(train_idx))

random_idx = list(range(total_train_img))
random.shuffle(random_idx)
total_train_img = np.array(total_train_img)[random_idx]
total_train_label = np.array(total_train_label)[random_idx]
total_train_sub = np.array(total_train_sub)[random_idx]

hf_train = h5py.File(save_path + 'train-train' + ".h5", 'w')
hf_train.create_dataset('img', data=total_train_img)
hf_train.create_dataset('lab', data=total_train_label)
hf_train.create_dataset('sub', data=total_train_sub)
hf_train.close()

random_idx = list(range(total_val_img))
random.shuffle(random_idx)
total_val_img = np.array(total_val_img)[random_idx]
total_val_label = np.array(total_val_label)[random_idx]
total_val_sub = np.array(total_val_sub)[random_idx]

hf_val = h5py.File(save_path + 'train-val' + ".h5", 'w')
hf_val.create_dataset('img', data=total_val_img)
hf_val.create_dataset('lab', data=total_val_label)
hf_val.create_dataset('sub', data=total_val_sub)

hf_val.close()