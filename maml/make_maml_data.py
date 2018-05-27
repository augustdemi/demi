import EmoData as ED
import glob
import cv2
import numpy as np

path = "/home/mihee/dev/project/robert_data/kshot/1/"



pp = ED.image_pipeline.FACE_pipeline(
        histogram_normalization=True,
        grayscale=True,
        rotation_range = 3,
        width_shift_range = 0.03,
        height_shift_range = 0.03,
        zoom_range = 0.03,
        random_flip = True,
        )



all_imgs = glob.glob(path + "*/*/*")

img_arr = []
save_path_arr = []
for img_file in all_imgs:
    img = cv2.imread(img_file)
    img_arr.append(img)
    save_path_arr.append(img_file)
img_arr, pts, pts_raw = pp.batch_transform(img_arr, preprocessing=True, augmentation= False)

for i in range(len(img_arr)):
    resized_img = cv2.resize(img_arr[i], (160,240))
    # cv2.imwrite("/home/mihee/dev/project/robert_code/data/test.jpg", resized_img * 255.0)
    cv2.imwrite(save_path_arr[i], resized_img * 255.0)

