# 实现调用摄像头、释放摄像头功能
# 思路：根据不同的源来调用摄像头或读取视频

import cv2
import time
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
# from PyQt5.QtCore import QTimer


class vid_source:
    def __init__(self):
        # self.timer_camera = QTimer()  # 初始化定时器
        self.cap = cv2.VideoCapture()  # 初始化摄像头
        self.vid_source = 0  # 初始化源，参数为0则调用笔记本内置摄像头，若需读取已有视频则将参数改为视频所在路径

    def set_vid_source(self, vid_source):
        self.vid_source = vid_source
        img = self.show_vid()
        return img

    def show_vid(self):
        self.flag, self.image = self.cap.read()  # 拿到flag 和帧
        print(type(self.image))
        self.show = cv2.flip(self.image, 1)  # 镜像翻转
        return self.flag, self.show

    def my_open(self):
        flag = self.cap.open(self.vid_source)
        return flag

    def my_close(self):
        self.cap.release()