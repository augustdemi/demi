import numpy as np
import os
import h5py
from sklearn.model_selection import KFold


############### img ############
path = "/home/ml1323/project/robert_data/DISFA/h5_3/"
# path = "D:/연구/프로젝트/Robert Walecki - deep_coder_submission_code/h5/"
from sklearn.model_selection import train_test_split



files = os.listdir(path)
file_idx = np.array(range(0,len(files)))
kf = KFold(n_splits=3)
kf.get_n_splits(file_idx)
k=3
train_index= list(range(0,18))
test_index = list(range(18,27))

#for train_index, test_index in kf.split(file_idx):
for j in range(0,1):
    print("TRAIN:", train_index, "TEST:", test_index)
    data_idx = {'train': train_index, 'test': test_index}
    for key in data_idx.keys():
        imgs = []
        labels =[]
        subjects =[]
        for i in data_idx[key]:
            file = files[i]
            print(">>>> file: " + file)
            hf = h5py.File(path+file, 'r')
            img=hf['img'].value[0:4844]
            label=hf['lab'].value[0:4844]
            hf.close()
            subject_number = int(file.split(".")[0].split("SN")[1])
            n_data = len(img)
            if(len(imgs) ==0):
                imgs = img
                labels = label
                subjects = np.ones(n_data) * subject_number
            else:
                imgs=np.vstack((imgs, img))
                labels=np.vstack((labels, label))
                subjects = np.vstack((subjects, np.ones(n_data) * subject_number))
        save_path = "/home/ml1323/project/robert_data/DISFA/kfold/" + str(k) + "/"
        if not os.path.exists(save_path):
                os.makedirs(save_path)
        # save_path = "D:/연구/프로젝트/Robert Walecki - deep_coder_submission_code/split/" + str(k) + "/"
        print(save_path + key + ".h5")
        subjects = subjects.reshape(len(imgs), )
        shuffled_img, dump_img, shuffled_label, dump_label, shuffled_sub, dump_sub = train_test_split(imgs, labels, subjects, test_size=0)

        hfs = h5py.File(save_path + key + ".h5", 'w')
        hfs.create_dataset('img', data=shuffled_img)
        hfs.create_dataset('lab', data=shuffled_label)
        hfs.create_dataset('sub', data=shuffled_sub)
        hfs.close()
    k+=1




