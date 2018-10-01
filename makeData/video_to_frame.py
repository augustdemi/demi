import cv2
import os

path = "/home/ml1323/project/robert_data/GET_FERA/SEMAINE-Sessions/SEMAINE-Sessions/"
save_path = "/home/ml1323/project/robert_data/FERA/semine/"
recs = [i.split(".")[0] for i in os.listdir(path) if i.endswith('.txt')]

for rec in recs:
    print(rec)
    file = [i for i in os.listdir(path + rec) if i.endswith('.avi')]
    vidcap = cv2.VideoCapture(path + rec + '/' + file)
    success, image = vidcap.read()
    count = 0
    success = True
    if not os.path.exists(save_path + rec):
        os.makedirs(save_path + rec)

    while success:
        cv2.imwrite(save_path + "/frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        # print('Read a new frame: ', success)
        count += 1
    print(count)
