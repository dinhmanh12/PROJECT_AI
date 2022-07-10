import cv2
import numpy as np

def checkParkingSpace(imgPro,imgResult):
    slot_list = []
    index_list_free = []
    index_list = []
    f = open("ImagePro_slot.csv","r")
    imgDilate,imgThreshold,imgMedian = PreProcess(imgResult)
    for index , line in enumerate(f.readlines()):
        slot = line.strip()
        x ,y ,w ,h = map(int,slot.split(","))
        slot_list.append([index+1,x,y,w,h])

    for slot in slot_list:
        index,x_s,y_s,w_s,h_s = slot
        imgCrop = imgPro[y_s:y_s+h_s,x_s:x_s+w_s]
        count = cv2.countNonZero(imgCrop)
        index_list.append(index)

        if count < 900:
            color = (0, 255, 0)
            thickness = 2
            index_list_free.append(index)
        else:
            color = (255, 255, 255)
            thickness = 2

        for slot in slot_list:
            cv2.rectangle (imgResult,(x_s,y_s),(x_s+w_s,y_s+h_s),color,thickness)
            cv2.putText(imgResult,f"{index}",(x_s,y_s),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),1)
            if count>900:
                cv2.rectangle (imgDilate,(x_s,y_s),(x_s+w_s,y_s+h_s),color,thickness)
    f = open("Status_ImagePro.csv","w+")
    f.write("slot,status\n")
    for i in index_list:
        if i in index_list_free:
            a = "Free"
        else:
            a = "Busy"
        f.write(f"{ i },{ a }\n")
    f.close()
    return imgDilate ,imgResult


def PreProcess(img):
    #img = cv2.resize(img, (1280, 720))
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(3,3),1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,25,16)
    imgMedian = cv2.medianBlur(imgThreshold,5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)
    return imgDilate,imgThreshold,imgMedian
