import numpy as np
import os
import h5py
from sklearn.model_selection import KFold


############### img ############
path = "/home/ml1323/project/robert_data/DISFA/h5_3/"
# path = "D:/연구/프로젝트/Robert Walecki - deep_coder_submission_code/h5/"
from sklearn.model_selection import train_test_split



files = os.listdir(path)

hf = h5py.File(path + "SN003.h5", 'r')
imgs = hf['img'].value[0:4844]
labels = hf['lab'].value[0:4844]

hf.close()
subject_number = 3
n_data = len(imgs)
subjects = np.ones(n_data) * subject_number

# save_path = "D:/연구/프로젝트/Robert Walecki - deep_coder_submission_code/split/" + str(k) + "/"
save_path = "/home/ml1323/project/robert_data/DISFA/sub3.h5"
#if not os.path.exists(save_path):
#    os.makedirs(save_path)

subjects = subjects.reshape(len(imgs), )
shuffled_img, dump_img, shuffled_label, dump_label, shuffled_sub, dump_sub = train_test_split(imgs, labels, subjects,
                                                                                              test_size=0)

hfs = h5py.File(save_path, 'w')
hfs.create_dataset('img', data=shuffled_img)
hfs.create_dataset('lab', data=shuffled_label)
hfs.create_dataset('sub', data=shuffled_sub)
hfs.close()
