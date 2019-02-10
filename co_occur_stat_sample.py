import numpy as np
import os

aus = ['au1', 'au2', 'au4', 'au6', 'au9', 'au12', 'au25', 'au26']

label_root_path = 'D:/연구/프로젝트/label/'
sample_info_path = 'D:/연구/프로젝트/sample_info/'

subjects = os.listdir(sample_info_path)
binary_intensity = lambda lab: 1 if lab > 0 else 0

ratio_mat = []

print(subjects)
for pau in aus:
    occur = {}
    for au in aus:
        occur.update({au: 0})

    for subject in subjects:
        print('=================================================================subject:', subject)
        label_paths = os.listdir(os.path.join(sample_info_path, subject))

        plabel_file = [path for path in label_paths if path.startswith(pau)][0]
        with open(os.path.join(sample_info_path, subject, plabel_file)) as f:
            num_on_int = int(plabel_file.split('on')[1].split('.')[0])
            on_intensity_frame_indices = [int(frame) + 1 for frame in f.readline().split(',')][:num_on_int]
            on_intensity_frame_indices = on_intensity_frame_indices[1:][::2]

        print('num of on_intensity: ', len(on_intensity_frame_indices))
        print(on_intensity_frame_indices)

        if len(on_intensity_frame_indices) == 0:
            continue
        occur[pau] += len(on_intensity_frame_indices)
        for au in aus:
            if au.startswith(pau):
                continue
            label_path = os.path.join(label_root_path, subject, subject + '_' + au + '.txt')
            with open(label_path) as f:
                lines = np.array(f.readlines())
                num_on_intensities = 0
                for line in lines:
                    if (int(line.split(",")[0]) in on_intensity_frame_indices) and (
                                int(line.split(",")[1].split("\n")[0]) > 0): num_on_intensities += 1
                occur[au] += num_on_intensities
                # print(occur)

    print(pau, '>>>>>>>> total:', occur)
    print('')
    print('')

    ratio = []
    for au in aus:
        ratio.append(occur[au])
    ratio = np.array(ratio)
    ratio_mat.append(ratio.round(decimals=2))
    print(ratio)

ratio_mat = np.array(ratio_mat)
print(ratio_mat)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
im = ax.imshow(ratio_mat, cmap="YlGnBu")

# We want to show all ticks...
ax.set_xticks(np.arange(len(aus)))
ax.set_yticks(np.arange(len(aus)))
# ... and label them with the respective list entries
ax.set_xticklabels(aus)
ax.set_yticklabels(aus)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(aus)):
    for j in range(len(aus)):
        text = ax.text(j, i, ratio_mat[i, j],
                       ha="center", va="center", color="r")

ax.set_title("num of co-occur frames given au_i(i=row) for few shot in training set.")
fig.tight_layout()
plt.show()
