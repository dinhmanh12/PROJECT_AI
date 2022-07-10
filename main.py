import streamlit as st
import cv2
from PIL import Image
import numpy as np
import PIL

from CNN_Process import *
from Image_Process import *


def main():

    st.title("BÃI ĐỖ XE THÔNG MINH")

    activities = ["CNN","Image Process"]
    choice = st.sidebar.selectbox("Select Activity",activities)

    if choice == 'CNN':
        img_file = st.file_uploader("Up file here",type=['jpg'])
        if img_file is not None:
            our_img = Image.open(img_file)
            st.header("Original Image")
            st.image(our_img)

        try:
            if img_file is not None:
                st.header("Process CNN")
                img_show = np.asarray(PIL.Image.open(img_file))
                img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
                img = Process_Image(our_img,img_show)
                st.image(img)
                path = 'Status_CNN.csv'
                csv_show(path)
        except:
            st.warning("SELECT IMAGE")

    if choice == 'Image Process':
        img_file = st.file_uploader("Up file here",type=['jpg'])
        if img_file is not None:
            our_img = Image.open(img_file)
            st.header("Original Image")
            st.image(our_img)
        
            st.header("Image Process")
            img_show = np.asarray(PIL.Image.open(img_file))
            img_show = cv2.resize(img_show, (1280, 720))

            st.header(" * Khử nhiễu")
            imgDilate,imgThreshold,imgMedian = PreProcess(img_show)
            col1,col2 = st.columns(2)
            with col1:
                st.image(imgThreshold,caption="Use Threshold")
            with col2:
                st.image(imgMedian,caption="Use Median")
            st.header("* Nhận diện")
            imgshow ,img= checkParkingSpace(imgDilate,img_show)
            col3,col4 = st.columns(2)
            with col3:
                st.image(imgDilate,caption="Use Dilate")
            with col4:
                st.image(imgshow,caption="Use countNonZero")
            st.header("Kết quả")
            st.image(img)
            path = 'Status_ImagePro.csv'
            csv_show(path)

def csv_show(path):
    st.header("STATUS PARKING")
    df = pd.read_csv(path)
    df = df.astype("string")
    new_df = df.T
    st.dataframe(new_df)

if __name__== '__main__':
    main()
    