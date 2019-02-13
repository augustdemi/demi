import numpy as np
import os

aus = ['au1', 'au2', 'au4', 'au6', 'au9', 'au12', 'au25', 'au26']

label_root_path = 'D:/연구/프로젝트/label/'

subjects = os.listdir(label_root_path)
binary_intensity = lambda lab: 1 if lab > 0 else 0

ratio_mat = []

# subjects = subjects[:2]
training_subjects = ['SN001', 'SN002', 'SN003', 'SN004', 'SN005', 'SN006', 'SN007', 'SN008', 'SN009', 'SN010', 'SN011',
                     'SN012', 'SN013', 'SN016']
subjects = training_subjects
# subjects = [sub for sub in subjects if sub not in training_subjects]
print(subjects)
for pau in aus:
    occur = {}
    for au in aus:
        occur.update({au: 0})

    for subject in subjects:
        # print('=================================================================subject:', subject)
        label_path = os.path.join(label_root_path, subject, subject + '_' + pau + '.txt')
        with open(label_path) as f:
            lines = np.array(f.readlines())
            # get the frame indices whose intensity is greater than 0
            on_intensity_frame_indices = [line.split(",")[0] for line in lines if
                                          int(line.split(",")[1].split("\n")[0]) > 0]

        # print('-- num of on_intensity: ', len(on_intensity_frame_indices))
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
                    # print(occur)

    print(pau, '>>>>>>>> total:', occur)

    ratio = []
    for au in aus:
        ratio.append(occur[au])
    ratio = np.array(ratio) / occur[pau]
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

ax.set_title("p(au_j=1 | au_i=1) for all samples in training set. (i =row, j=column)")
fig.tight_layout()
plt.show()
