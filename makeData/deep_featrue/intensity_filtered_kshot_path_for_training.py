import os

original_frame_path = "/home/ml1323/project/robert_data/DISFA/detected_disfa/"
save_path = "/home/ml1323/project/robert_data/DISFA/kshot_path_filter_8au/train/"

all_au = ['au1', 'au2', 'au4', 'au6', 'au9', 'au12', 'au25', 'au26']
train_subjects = ['SN001', 'SN002', 'SN003', 'SN004', 'SN005', 'SN006', 'SN007', 'SN008', 'SN009', 'SN010', 'SN011',
                  'SN012', 'SN013', 'SN016']
cnt = 0
cnta_arr = []

for au in all_au:
    for subject in os.listdir(original_frame_path):
        detected_img_files = os.listdir(original_frame_path + subject)
        detected_frame_idx = [int(elt.split('frame')[1].split('_')[0]) for elt in detected_img_files]
        detected_frame_idx = list(set(detected_frame_idx))

        detected_frame_idx.sort()
        all_intensities = [0] * (detected_frame_idx[-1] + 1)
        #### filter ####
        for au in all_au:
            label_path = "/home/ml1323/project/robert_data/DISFA/label/" + subject + "/" + subject + "_" + au + ".txt"
            f = open(label_path, 'r')
            all_labels = f.readlines()
            # print(all_labels)

            for idx in detected_frame_idx:
                # print(int(all_labels[idx].split(",")[1].split("\n")[0]))
                try:
                    intensity = int(all_labels[idx].split(",")[1].split("\n")[0])
                    all_intensities[idx] += intensity
                except:
                    # print("out of index: ", idx, " for au, subject: ", au, subject)
                    continue
        detected_frame_idx = [idx for idx in range(len(all_intensities)) if all_intensities[idx] > 6]
        cnt += len(detected_frame_idx)
        cnta_arr.append(len(detected_frame_idx))

        label_path = "/home/ml1323/project/robert_data/DISFA/label/" + subject + "/" + subject + "_" + au + ".txt"
        intensities_for_one_au = []
        f = open(label_path, 'r')
        all_labels = f.readlines()
        train_on_idx = []
        train_off_idx = []
        for idx in detected_frame_idx:
            try:
                intensity = int(all_labels[idx].split(",")[1].split("\n")[0])
            except:
                print("out of index: ", idx, " for au, subject: ", au, subject)
                continue
            if intensity > 0:
                train_on_idx.append(idx)
            else:
                train_off_idx.append(idx)

        print('train_on_idx len: ', len(train_on_idx))
        print('train_off_idx len: ', len(train_off_idx))

        save_path_per_au_sub = save_path + au + "/" + subject
        if not os.path.exists(save_path_per_au_sub + "/on"): os.makedirs(save_path_per_au_sub + "/on")
        if not os.path.exists(save_path_per_au_sub + "/off"): os.makedirs(save_path_per_au_sub + "/off")

        # copy on intensity frames for train
        file_path_to_save = []
        with open(save_path_per_au_sub + "/on/file_path.csv", 'w') as f:
            for i in train_on_idx:
                file_path_to_save.append(original_frame_path + subject + "/frame" + str(i) + "_0.jpg")
            f.write(','.join(file_path_to_save))

        # copy off intensity frames for train
        file_path_to_save = []
        with open(save_path_per_au_sub + "/off/file_path.csv", 'w') as f:
            for i in train_off_idx:
                file_path_to_save.append(original_frame_path + subject + "/frame" + str(i) + "_0.jpg")
            f.write(','.join(file_path_to_save))
        print(">>>>> done: ", subject)
    print("========================================= done: ", au)
print('cnt:', cnt)
print('cnta_arr: ', cnta_arr)
