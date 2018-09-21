import pickle
import os
import numpy as np

path = "/home/ml1323/project/robert_data/DISFA/detected_disfa/"
save_path = "/home/ml1323/project/robert_data/DISFA/new_dataset/trainset/"
if not os.path.exists(save_path): os.makedirs(save_path)

subjects = os.listdir(path)
subjects.sort()
all_au = ['au1', 'au2', 'au4', 'au5', 'au6', 'au9', 'au12', 'au15', 'au17', 'au20', 'au25', 'au26']

# file_len = []
# for subject in subjects:
#     subject_folder = os.path.join(path, subject)
#     files = os.listdir(subject_folder)
#     file_len.append(len(files))
#
# print('original files len : ', file_len)
# min_file_len = min(file_len)

for subject in subjects:
    subject_folder = os.path.join(path, subject)
    files = os.listdir(subject_folder)
    test_file_names = [os.path.join(subject_folder, file) for file in files]
    print('original files len in ', subject, ' : ', len(files))
    frame_idx = [int(file.split('_')[0].split('frame')[1].split('.')[0]) for file in files]

    label_per_subject = []
    for au in all_au:
        label_path = "/home/ml1323/project/robert_data/DISFA/label/" + subject + "/" + subject + "_" + au + ".txt"
        intensities_for_one_au = []
        f = open(label_path, 'r')
        all_labels = f.readlines()

        label_per_au = []
        for idx in frame_idx:
            try:
                intensity = int(all_labels[idx].split(",")[1].split("\n")[0])
            except:
                print("on_idx - out of index: ", idx, " for au, subject: ", au, subject)
                continue
            if intensity > 0:
                label_per_au.append([0, 1])
            else:
                label_per_au.append([1, 0])
        # label_per_au = (4800,2)
        label_per_subject.append(label_per_au)  # (12, 4800,2)
    label_per_subject = np.array(label_per_subject)
    print('label_per_subject shape: ', label_per_subject.shape)
    print(label_per_subject[:3])
    label_per_subject = np.transpose(label_per_subject, (1, 0, 2))  # (4800,12,2)
    print('label_per_subject shape after : ', label_per_subject.shape)
    print(label_per_subject[:3])
    print('>>>>>>>>>>>>>>>>>>>>>>>> sub done:', subject)

    np.random.seed(3)
    np.random.shuffle(test_file_names)
    np.random.seed(3)
    np.random.shuffle(label_per_subject)
    out = open(save_path + subject + ".pkl", 'wb')
    pickle.dump({'test_file_names': test_file_names, 'lab': label_per_subject}, out, protocol=2)
