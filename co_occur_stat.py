import numpy as np
import os

aus = ['au1', 'au2', 'au4', 'au6', 'au9', 'au12', 'au25', 'au26']

label_root_path = '/home/ml1323/project/robert_data/DISFA/label/'

subjects = os.listdir(label_root_path)
binary_intensity = lambda lab: 1 if lab > 0 else 0

occur = {}
for au in aus:
    occur.update({au: 0})

for subject in subjects:
    print('=================================================================subject:', subject)
    pau = 'au1'
    label_path = os.path.join(label_root_path, subject, subject + '_' + pau + '.txt')
    with open(label_path) as f:
        lines = np.array(f.readlines())
        # get the frame indices whose intensity is greater than 0
        on_intensity_frame_indices = [line.split(",")[0] for line in lines if
                                      int(line.split(",")[1].split("\n")[0]) > 0]

    print('-- num of on_intensity: ', len(on_intensity_frame_indices))
    # print(on_intensity_frame_indices)
    occur[pau] += len(on_intensity_frame_indices)
    for au in aus:
        if au.startswith(pau):
            continue
        else:
            label_path = os.path.join(label_root_path, subject, subject + '_' + au + '.txt')
            with open(label_path) as f:
                lines = np.array(f.readlines())
                num_on_intensities = 0
                for line in lines:
                    if (line.split(",")[0] in on_intensity_frame_indices) and (
                        int(line.split(",")[1].split("\n")[0]) > 0): num_on_intensities += 1
                occur[au] += num_on_intensities
    print(occur)

print('>>>>>>>> total:', occur)

ratio = np.array(list(occur.values())) / occur[pau]
print(ratio)
