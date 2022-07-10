from operator import index
import cv2

img = cv2.imread("image_c9\C9_base_27.01.2021_13.40.23.jpg")
img = cv2.resize(img, (1280, 720))
rois = cv2.selectROIs("image", img, showCrosshair=False, fromCenter=False)
f = open("ImagePro_slot.csv", "w+")
f.write("x,y,w,h,index\n")
index = 0
for roi in rois:
    index +=1
    x, y, w, h = roi
    f.write(f"{x},{y},{w},{h},{index}\n")
f.close()