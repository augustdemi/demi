import cv2
import numpy as np
path = "/home/ml1323/project/robert_data/DISFA/video/RightVideoSN002_comp.avi"
vidcap=cv2.VideoCapture(path)
success,image = vidcap.read()
count = 0
success = True
while success:
  cv2.imwrite("SN002/frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
print(count)
