import random
import os

original_frame_path = "/home/ml1323/project/robert_data/DISFA/detected_disfa/"
save_path = "/home/ml1323/project/robert_data/DISFA/kshot_path/"
all_au = ['au1', 'au2', 'au4', 'au6', 'au9', 'au12', 'au25', 'au26']
train_subjects = ['SN001', 'SN002', 'SN003', 'SN004', 'SN005', 'SN006', 'SN007', 'SN008', 'SN009', 'SN010', 'SN011',
                  'SN012', 'SN013', 'SN016']

for subject in train_subjects:
    detected_img_files = os.listdir(original_frame_path + subject)
    detected_frame_idx = [int(elt.split('frame')[1].split('_')[0]) for elt in detected_img_files]
    detected_frame_idx = list(set(detected_frame_idx))

    file_path_to_save = []
    # copy test_b
    if not os.path.exists(save_path + "validation/" + subject): os.makedirs(save_path + "validation/" + subject)
    with open(save_path + "validation/" + subject + "/file_path.csv", 'w') as f:
        for i in detected_frame_idx:
            file_path_to_save.append(original_frame_path + subject + "/frame" + str(i) + "_0.jpg")
        f.write(','.join(file_path_to_save))
