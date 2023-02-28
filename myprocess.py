# 主函数，完成所有类的调用，和完成所有界面的操作
# import time
from PyQt5.QtCore import QTimer
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from inter_main61 import Ui_MainWindow
# from login import Ui_LoginMain
from video_cam import vid_source
import cv2
import sys
from detect import detectImg
import threading
import winsound
from PyQt5.QtWidgets import QApplication, QHBoxLayout
from PyQt5.QtWebEngineWidgets import *
from my_echart import *
from save_pic import ImageSave
from sql_db import Db


class MainMain(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainMain, self).__init__(parent)
        self.setupUi(self)
        self.initUI()
        # self.login_win = Ui_LoginMain()
        self.thread_time()
        self.timer_video = QTimer()
        self.timer_camera = QTimer()  # 初始化定时器
        self.timer_chart = QTimer()
        self.cap = cv2.VideoCapture()  # 初始化摄像头，参数为0则调用笔记本内置摄像头，若需读取已有视频则将参数改为视频所在路径
        self.slot_init()
        self.myvid = vid_source()
        self.img = None
        self.det = detectImg()
        self.device = self.det.opt.device
        self.det_flag = True
        self.cur_img = None
        self.vid_ret = True
        self.mychart = charts()
        self.nomask_rate_list = []
        self.mask_rate_list = []
        self.save_pic = ImageSave()
        self.db = Db()
        self.pic_index = 1
        self.pic_list_len = None
        self.save_path_text.setText(self.save_pic.path)
        self.det_device_text.setText(self.device)  # 获取运行设备

    # 把按钮和函数对应起来
    def initUI(self):
        print("把按钮和函数对应起来")
        self.btn_open_img.clicked.connect(self.open_img)  # 绑定导入图片控件
        self.btn_detect_img.clicked.connect(self.det_img)  # 图片检测
        self.btn_open_video.clicked.connect(self.open_video)  # 绑定导入视频控件
        self.btn_close_video.clicked.connect(self.pause_det_vid)  # 停止视频检测并清空视频
        self.btn_open_camera.clicked.connect(self.slotCameraButton)  # 绑定打开相机控件
        self.listWidget.itemClicked.connect(self.item_clicked)  # 将系统设置下的左侧菜单栏按钮和右侧界面连接起来
        self.btn_close_win.clicked.connect(self.closeEvent)  # 自定义关闭按钮触发
        self.btn_change_path.clicked.connect(self.change_pic_path)  # 更改存图路径
        self.btn_left.clicked.connect(self.btn_left_pic)  # 查看统计页面左向按钮触发
        self.btn_right.clicked.connect(self.btn_right_pic)  # 查看统计页面右向按钮触发
        self.warn_time_combo.currentIndexChanged.connect(self.stat_thread)  # 当选择信息来源触发此线程-》用于查看检测图片

    # 系统设置下的左侧菜单栏按钮和右侧界面连接起来
    def item_clicked(self):
        item = self.listWidget.currentItem().text()
        # print('item', item)
        if item == '更改配置':
            self.stackedWidget.setCurrentIndex(0)
        elif item == '查看统计':
            self.stackedWidget.setCurrentIndex(1)

    # 线程-》用于查看检测图片
    def stat_thread(self):
        stat_thread = threading.Thread(target=self.stat_func)
        stat_thread.setDaemon(True)
        stat_thread.start()

    def slot_init(self):
        self.timer_camera.timeout.connect(self.showImage)  # 计时器timeout后执行操作 用于打开摄像头

    # 拿到摄像头的帧，送去检测然后显示到界面上
    def showImage(self):
        self.det_flag = True
        if self.det_flag:
            self.flag, self.cam_img = self.myvid.show_vid()
            if self.flag:
                outImg, pred, names, nomask_rate_list, mask_rate_list, self.classes_list = self.det.predictImage(self.cam_img)
                print('11111')
                showImage = self.trans_format(outImg)
                self.cam_label.setPixmap(QPixmap.fromImage(showImage))
                if 0 in self.classes_list:
                    save_pic_thread = threading.Thread(target=self.save_pic.save_detect_img, kwargs={'img': outImg, 'pic_source': '0'})
                    save_pic_thread.setDaemon(True)
                    save_pic_thread.start()
                cam_chart_thread = threading.Thread(target=self.mychart.make_chart(nomask_rate_list, mask_rate_list))
                cam_chart_thread.setDaemon(True)
                cam_chart_thread.start()
                self.open_cam_chart_display()
                tip_thread = threading.Thread(target=self.warning_sound)
                tip_thread.setDaemon(True)
                tip_thread.start()

    # 触发打开相机的条件，打开关闭摄像头控制
    def slotCameraButton(self):
        if not self.timer_camera.isActive():
            self.flag = self.myvid.my_open()
            if not self.flag:
                msg = QMessageBox.Warning(self, u'Warning', u'请检测相机与电脑是否连接正确',
                                          buttons=QMessageBox.Ok,
                                          defaultButton=QMessageBox.Ok)
            else:
                self.timer_camera.start(30)
                self.btn_open_camera.setText('关闭摄像头实时监测')
        else:
            # 关闭摄像头并清空显示信息
            self.timer_camera.stop()
            self.myvid.my_close()
            self.cam_label.clear()
            self.label_2.clear()
            self.classes_list = []
            self.btn_open_camera.setText('打开摄像头实时监测')

    # 将 opencv 格式的图片转换为 qimage 格式的图片
    def trans_format(self, img):
        show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)  # mat --> qimage
        return showImage

    # 导入图片
    def open_img(self):
        self.img_label.clear()
        self.det_img_result.clear()
        # 弹出一个文件选择框
        print("jinzhe")
        imgName, imgType = QFileDialog.getOpenFileName(self, "导入图片", "", "*.jpg;;*.png;;All File(*)")    # imgName 拿到的图片
        print("imgName", imgName)
        self.cur_pic = imgName
        print(self.cur_pic)
        img = QtGui.QPixmap(imgName).scaled(self.img_label.width(), self.img_label.height())
        self.img_label.setPixmap(img)

    # 图片检测
    def det_img(self):
        self.cur_pic = cv2.imread(self.cur_pic)
        print("self.cur_pic", type(self.cur_pic))
        outImg, pred, names, nomask_rate_list, mask_rate_list, classes_list = self.det.predictImage(self.cur_pic)
        showImage = self.trans_format(outImg)
        img = QtGui.QPixmap(showImage)
        self.det_img_result.setPixmap(img)

    # 导入视频
    def open_video(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, "导入视频", "", "*.mp4 *.avi")
        if fileName:
            print(fileName)
            self.cap = cv2.VideoCapture(fileName)
            self.timer_video.start(30)
            self.timer_video.timeout.connect(self.openvid)

    # 视频检测
    def openvid(self):
        if(self.cap.isOpened()):
            self.vid_ret, self.img = self.cap.read()      # self.img 拿到视频的帧
            print(self.vid_ret)
            print(type(self.img))
            if self.vid_ret:
                outImg, pred, names, nomask_rate_list, mask_rate_list, self.classes_list = self.det.predictImage(self.img)
                showImage = self.trans_format(outImg)
                self.vid_label.setPixmap(QPixmap.fromImage(showImage))
                if 0 in self.classes_list:
                    save_vid_pic_thread = threading.Thread(target=self.save_pic.save_detect_img, kwargs={'img': outImg, 'pic_source': '1'})
                    save_vid_pic_thread.setDaemon(True)
                    save_vid_pic_thread.start()
                print("===========")
                if len(nomask_rate_list) == 0 and len(mask_rate_list) == 0:
                    pass
                else:
                    vid_chart_thread = threading.Thread(target=self.mychart.make_chart(nomask_rate_list, mask_rate_list))
                    vid_chart_thread.setDaemon(True)
                    vid_chart_thread.start()
                self.open_vid_chart_display()
                tip_thread = threading.Thread(target=self.vid_warning_sound)
                tip_thread.setDaemon(True)
                tip_thread.start()
            else:
                self.cap.release()
                self.timer_video.stop()
                self.nomask_rate_list = []
                self.mask_rate_list = []

    # 停止视频检测 并清空label
    def pause_det_vid(self):
        self.cap.release()
        self.timer_video.stop()
        self.vid_ret = False
        self.vid_label.clear()
        self.result_label.clear()
        self.nomask_rate_list = []
        self.mask_rate_list = []

    # 获取实时时间和日期的线程
    def thread_time(self):
        my_time_thread = threading.Thread(target=self.statushowtime())
        my_time_thread.setDaemon(True)
        my_time_thread.start()

    # 显示时间和日期
    def showtime(self):
        my_time = QDateTime.currentDateTime()  # 获取系统当前时间
        self.time_date = my_time.toString('yyyy-MM-dd')
        self.time_time = my_time.toString('hh:mm:ss')
        self.date_text.setText(self.time_date)
        self.time_text.setText(self.time_time)

    # 触发更新时间和日期的条件
    def statushowtime(self):
        self.timer_time = QTimer()
        self.timer_time.timeout.connect(self.showtime)
        self.timer_time.start(1)  # 每隔一秒刷新一次

    # # 触发更新echart
    # def cam_show_echart(self):
    #     self.timer_cam_chart = QTimer()
    #     self.timer_cam_chart.timeout.connect(self.open_cam_chart_display)
    #     self.timer_cam_chart.start(0.1)

    # 显示视频检测的置信度折线图
    def open_vid_chart_display(self):
        self.line_grid = QGridLayout()
        self.brower = QWebEngineView()
        self.brower.setGeometry(10, 320, 301, 201)
        self.line_grid.addWidget(self.brower)
        self.brower.load(QUrl(QFileInfo("./line_nomask_mask.html").absoluteFilePath()))
        self.frame_video.setLayout(self.line_grid)

    # 显示摄像头监测的置信度折线图
    def open_cam_chart_display(self):
        self.line_grid2 = QGridLayout()
        self.brower2 = QWebEngineView()
        self.brower2.setGeometry(12, 313, 301, 201)
        self.line_grid2.addWidget(self.brower2)
        self.brower2.load(QUrl(QFileInfo("./line_nomask_mask.html").absoluteFilePath()))
        self.frame_camera.setLayout(self.line_grid2)

    # 摄像头语音警报功能
    def warning_sound(self):
        if 0 in self.classes_list:
            self.label_2.setText("检测到未佩戴口罩人员！")
            duration = 1000  # 持续时间 单位是毫秒
            freq = 440  # 频率 单位是赫兹
            winsound.Beep(freq, duration)
            winsound.PlaySound('Tik Tok.wav', winsound.SND_FILENAME | winsound.SND_NODEFAULT)
        else:
            self.label_2.clear()
            winsound.PlaySound("Tik Tok.wav", winsound.SND_PURGE | winsound.SND_NODEFAULT)

    # 视频语音警报功能
    def vid_warning_sound(self):
        if 0 in self.classes_list:
            self.result_label.setText("检测到未佩戴口罩人员！")
            duration = 1000  # 持续时间 单位是毫秒
            freq = 440  # 频率 单位是赫兹
            winsound.Beep(freq, duration)
            winsound.PlaySound('Tik Tok.wav', winsound.SND_FILENAME | winsound.SND_NODEFAULT)
        else:
            self.result_label.clear()
            winsound.PlaySound("Tik Tok.wav", winsound.SND_PURGE | winsound.SND_NODEFAULT)

    # 点击左切按钮
    def btn_left_pic(self):
        if self.pic_index == 0:
            self.btn_left.setEnabled(False)
            self.btn_right.setEnabled(True)
            print("=====================", self.pic_index)
        else:
            self.pic_index -= 1
            print("=====================", self.pic_index)
        pic_path = self.pic_cam_path_list[self.pic_index]
        warn_time = self.db.pic_warn_time(pic_path)
        print(warn_time)
        warn_info_chart = "检测到未佩戴口罩人员！    --" + warn_time
        self.warn_info_text.setText(warn_info_chart)
        pixmap = QtGui.QPixmap(pic_path)
        self.warn_pic.setPixmap(pixmap)
        self.warn_pic.setScaledContents(True)

    # 点击右切按钮
    def btn_right_pic(self):
        if self.pic_index >= self.pic_list_len:
            self.btn_right.setEnabled(False)
            self.btn_left.setEnabled(True)
            self.pic_index = 1
            print("=====================", self.pic_index)
        elif self.pic_index < (self.pic_list_len - 1):
            self.pic_index += 1
            print("=====================", self.pic_index)
        pic_path = self.pic_cam_path_list[self.pic_index]
        warn_time = self.db.pic_warn_time(pic_path)
        print(warn_time)
        warn_info_chart = "检测到未佩戴口罩人员！    --" + warn_time
        self.warn_info_text.setText(warn_info_chart)
        pixmap = QtGui.QPixmap(pic_path)
        self.warn_pic.setPixmap(pixmap)
        self.warn_pic.setScaledContents(True)

    # 系统设置模块
    def stat_func(self):
        self.pic_cam_path_list = []
        self.pic_index = 0
        self.warn_pic.clear()
        source_text = self.warn_time_combo.currentText()
        self.pic_cam_path_list = self.db.select_pic_info(source_text)
        # self.warn_info_text.setText("检测到未佩戴口罩人员！")
        self.pic_list_len = len(self.pic_cam_path_list)
        print("------------------self.pic_list_len", self.pic_list_len)
        # while self.pic_index <= len(self.pic_cam_path_list):
        #     print("++++++++++++++++++++", self.pic_index)
        #     pic_path = self.pic_cam_path_list[self.pic_index]
        #     pixmap = QtGui.QPixmap(pic_path)
        #     self.warn_pic.setPixmap(pixmap)
        #     self.warn_pic.setScaledContents(True)
        pass

    # 判断是否包含中文字符
    def is_chinese(self, path):
        for ch in path:
            if u'\u4e00' <= ch <= u'\u99ff':
                return True
        return False

    # 更改保存图片的路径
    def change_pic_path(self):
        file_path = QFileDialog.getExistingDirectory(self, "选择文件夹", "/")
        if file_path:
            if self.is_chinese(file_path):
                QMessageBox.about(self, '提示', '路径中存在中文字符！请重新选择存图文件夹。')
                self.save_path_text.setText('G:/data')
            else:
                self.save_pic.set_path(file_path)
                self.save_path_text.setText(file_path)
        else:
            pass

    # 界面关闭触发
    def closeEvent(self):
        reply = QMessageBox.question(self,
                                     '注意',
                                     "确认退出系统？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            # self.login_win.show()
            # event.accept()
        else:
            # event.ignore()
            pass


if __name__ == "__main__":
    app = QApplication(sys.argv)  # 固定的，表示程序应用
    myWin = MainMain()  # 实例化
    myWin.show()
    sys.exit(app.exec_())  # 不加这句，程序界面会一闪而过
