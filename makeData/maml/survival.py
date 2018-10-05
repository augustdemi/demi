import os
import pickle
from shutil import copyfile

save_path = "/home/ml1323/project/robert_data/DISFA/neww_dataset/"
lable_dir = '/home/ml1323/project/robert_data/DISFA/label/'
subjects = os.listdir(lable_dir)

original_frame_path = "/home/ml1323/project/robert_data/DISFA/detected_disfa/"
all_au = ['au1', 'au2', 'au4', 'au5', 'au6', 'au9', 'au12', 'au15', 'au17', 'au20', 'au25', 'au26']
test_subjects = ['SN017', 'SN018', 'SN021', 'SN023', 'SN024', 'SN025', 'SN026', 'SN027', 'SN028', 'SN029', 'SN030',
                 'SN031', 'SN032']

for subject in test_subjects:
    data = pickle.load(open('/home/ml1323/project/robert_data/DISFA/new_dataset/testset/' + subject + '.pkl', "rb"),
                       encoding='latin1')
    test_a_idx = [x.split('/')[9].split('.')[0].split('frame')[1] for x in data['test_file_names']]
    print(subject)
    print(test_a_idx)

    for au in all_au:
        label_path = "/home/ml1323/project/robert_data/DISFA/label/" + subject + "/" + subject + "_" + au + ".txt"
        intensities_for_one_au = []
        f = open(label_path, 'r')
        all_labels = f.readlines()

        test_a_on_idx = []
        test_a_off_idx = []
        for idx in test_a_idx:
            try:
                intensity = int(all_labels[idx].split(",")[1].split("\n")[0])
            except:
                print("test_a_on_idx - out of index: ", idx, " for au, subject: ", au, subject)
                continue
            if intensity > 0:
                test_a_on_idx.append(idx)
            else:
                test_a_off_idx.append(idx)

        print('test_a_on_idx len: ', len(test_a_on_idx))
        print('test_a_off_idx len: ', len(test_a_off_idx))

        if not os.path.exists(save_path + "/test_a/" + au + "/" + subject + "/on"): os.makedirs(
            save_path + "/test_a/" + au + "/" + subject + "/on")
        if not os.path.exists(save_path + "/test_a/" + au + "/" + subject + "/off"): os.makedirs(
            save_path + "/test_a/" + au + "/" + subject + "/off")

        # copy on intensity frames for test_a
        for i in test_a_on_idx:
            copyfile(original_frame_path + subject + "/frame" + str(i) + "_0.jpg",
                     save_path + "/test_a/" + au + "/" + subject + "/on/frame" + str(i) + ".jpg")

        # copy off intensity frames for test_a
        for i in test_a_off_idx:
            copyfile(original_frame_path + subject + "/frame" + str(i) + "_0.jpg",
                     save_path + "/test_a/" + au + "/" + subject + "/off/frame" + str(i) + ".jpg")
