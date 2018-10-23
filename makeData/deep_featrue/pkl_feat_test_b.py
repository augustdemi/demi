import pickle
import os
import numpy as np

kshot_path = "/home/ml1323/project/robert_data/DISFA/kshot_path/test_b/"
feat_path = "/home/ml1323/project/robert_data/DISFA/kshot_path/test_b/"
save_path = "/home/ml1323/project/robert_data/DISFA/kshot_path/testset/"
if not os.path.exists(save_path): os.makedirs(save_path)

subjects = os.listdir(kshot_path)
subjects.sort()
all_au = ['au1', 'au2', 'au4', 'au6', 'au9', 'au12', 'au25', 'au26']

for subject in subjects:
    subject_folder = os.path.join(path, subject)
    files = os.listdir(subject_folder)
    test_file_paths = open(os.path.join(path, subject, 'file_path.csv'))
    print('original files len in ', subject, ' : ', len(files))
    frame_idx = [int(path.split('/')[-1].split('.')[0].split('_')[0]) for path in test_file_paths]

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
    np.random.shuffle(test_file_paths)
    np.random.seed(3)
    np.random.shuffle(label_per_subject)

    test_features = []
    # 모든 feature 파일이 존재하는 경로
    feature_file_path = feat_path + '/' + subject + '.csv'
    print('=============feature_file_path: ', feature_file_path)
    f = open(feature_file_path, 'r')
    lines = f.readlines()
    all_feat_data = {}  # 모든 feature를 frame 을 key값으로 하여 dic에 저장해둠
    for line in lines:
        line = line.split(',')
        all_feat_data.update({line[1]: line[2:]})  # key = frame, value = feature vector

    for path in test_file_paths:
        try:
            frame = path.split('/')[-1].split('.')[0].split('_')[0]
            test_features.append(all_feat_data[frame])
        except:
            print('CHECK DATA FOR frame: ', frame, ' from ', path)
    print('len of test_features:', test_features)

    out = open(save_path + subject + ".pkl", 'wb')
    pickle.dump({'test_file_names': test_features, 'lab': label_per_subject}, out, protocol=2)
