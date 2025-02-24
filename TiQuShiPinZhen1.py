


'''
提取单个视频的所有帧
'''
import cv2
import numpy as np
def save_image(image,addr,num):
    #存储的图片路径
    address=addr+str(num)+'.jpg'
    #存储图片
    cv2.imwrite(address,image)
#读入视频
videoCapture=cv2.VideoCapture("G:/Code/Matalab/ColorTransfer/video/park.avi")
#读取视频帧
success,frame=videoCapture.read()
i=0
while success:
    i=i+1
    #保存图片
    # save_image(frame,'Images/photo/Images',i)
    save_image(frame, 'G:/Code/Matalab/ColorTransfer/Video_Images_Park/', i)
    if success:
        print('save image:',i)
    #读取视频帧
    sucess,frame=videoCapture.read()

