import cv2
import os
import numpy as np
import h5py

path = "/home/ml1323/project/robert_data/MNIST/trainingSet/"
hf_path = "/home/ml1323/project/robert_data/MNIST/h5/mnist.h5"

with h5py.File(hf_path, "a") as hf:
    for i in range(0,10):
        dir = path + str(i)
        files = os.listdir(dir)
        img_data = []
        label_data = []
        for file in files:
            img = cv2.imread(dir + "/" + file)
            resized_img = cv2.resize(img, (160, 240))
            img_data.append(resized_img)
            one_label_onehot = np.zeros(10)
            one_label_onehot[i] = 1
            label_data.append(one_label_onehot)
        img_data = np.array(img_data)
        label_data = np.array(label_data)
        print(img_data.shape)
        print(label_data[0])
        if(i==0):
            hf.create_dataset('img', data=img_data,chunks=True, maxshape=(None,240,160,3))
            hf.create_dataset('lab', data=label_data, chunks= True, maxshape=(None,10))
            hf.create_dataset('s', data=np.array([1,2]))
        else:
            hf['img'].resize((hf['img'].shape[0] + img_data.shape[0]), axis=0)
            hf['lab'].resize((hf['lab'].shape[0] + label_data.shape[0]), axis=0)
            hf['img'][-img_data.shape[0]:] = img_data
            hf['lab'][-label_data.shape[0]:] = label_data
        print(">>>>>>>>>>> end i = " + str(i) )


hf.close()


