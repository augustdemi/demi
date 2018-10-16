import random
import os

original_frame_path = "/home/ml1323/project/robert_data/DISFA/detected_disfa/"
save_path = "/home/ml1323/project/robert_data/DISFA/kshot_path/"
all_au = ['au1', 'au2', 'au4', 'au6', 'au9', 'au12', 'au25', 'au26']
test_subjects = ['SN017', 'SN018', 'SN021', 'SN023', 'SN024', 'SN025', 'SN026', 'SN027', 'SN028', 'SN029', 'SN030',
                 'SN031', 'SN032']

for subject in test_subjects:
    detected_img_files = os.listdir(original_frame_path + subject)
    detected_frame_idx = [int(elt.split('frame')[1].split('_')[0]) for elt in detected_img_files]
    detected_frame_idx = list(set(detected_frame_idx))
    random.seed(1)
    test_a_idx = random.sample(detected_frame_idx, int(len(detected_frame_idx) / 2))
    test_b_idx = [i for i in detected_frame_idx if i not in test_a_idx]
    print('test_a len: ', len(test_a_idx))
    print('test_b len: ', len(test_b_idx))

    ########### 위의 각 subject 별 fixed test_a, test_b에 대해서, 각 au별로 on/off를 구해 분리해둠.
    ########### 그리곤 test_b셋에대해서 testset pickle값을 만들어 둬서 이 전체 데이터 셋에 대해서 evaluation할 것임.

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

        test_b_on_idx = []
        test_b_off_idx = []
        for idx in test_b_idx:
            try:
                intensity = int(all_labels[idx].split(",")[1].split("\n")[0])
            except:
                print("test_b_on_idx - out of index: ", idx, " for au, subject: ", au, subject)
                continue
            if intensity > 0:
                test_b_on_idx.append(idx)
            else:
                test_b_off_idx.append(idx)
        f.close()

        print('test_a_on_idx len: ', len(test_a_on_idx))
        print('test_a_off_idx len: ', len(test_a_off_idx))
        print('test_b_on_idx len: ', len(test_b_on_idx))
        print('test_b_off_idx len: ', len(test_b_off_idx))

        save_path_teat_a_per_au_sub = save_path + "/test_a/" + au + "/" + subject

        if not os.path.exists(save_path_teat_a_per_au_sub + "/on"): os.makedirs(save_path_teat_a_per_au_sub + "/on")
        if not os.path.exists(save_path_teat_a_per_au_sub + "/off"): os.makedirs(save_path_teat_a_per_au_sub + "/off")

        # copy on intensity frames for train
        file_path_to_save = []
        with open(save_path_teat_a_per_au_sub + "/on/file_path.csv", 'w') as f:
            for i in test_a_on_idx:
                file_path_to_save.append(original_frame_path + subject + "/frame" + str(i) + "_0.jpg")
            f.write(','.join(file_path_to_save))

        # copy off intensity frames for train
        file_path_to_save = []
        with open(save_path_teat_a_per_au_sub + "/on/file_path.csv", 'w') as f:
            for i in test_a_off_idx:
                file_path_to_save.append(original_frame_path + subject + "/frame" + str(i) + "_0.jpg")
            f.write(','.join(file_path_to_save))
        print(">>>>> done: ", subject)
    print("========================================= done: ", au)

    # copy test_b
    if not os.path.exists(save_path + "test_b/" + subject): os.makedirs(save_path + "test_b/" + subject)
    with open(save_path + "test_b/" + subject + "/file_path.csv", 'w') as f:
        for i in test_b_idx:
            file_path_to_save.append(original_frame_path + subject + "/frame" + str(i) + "_0.jpg")
        f.write(','.join(file_path_to_save))
