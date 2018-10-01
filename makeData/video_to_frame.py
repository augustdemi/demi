import cv2
import os
import numpy as np

path = "/home/ml1323/project/robert_data/GET_FERA/SEMAINE-Sessions/SEMAINE-Sessions/rec1/2008.12.05.16.03.15_User Frontal_C_Obadiah.avi"
vidcap=cv2.VideoCapture(path)
success,image = vidcap.read()
count = 0
success = True
save_path = "/home/ml1323/project/robert_data/FERA/semine/rec1/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
while success:
    cv2.imwrite(save_path + "frame%d.jpg" % count, image)  # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
print(count)
