# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'inter_main6.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5.QtCore import Qt, QSize
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QCursor


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1412, 830)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        MainWindow.setUnifiedTitleAndToolBarOnMac(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(0, 0, 1412, 806))
        self.frame.setStyleSheet("QWidget{\n"
"border-top-color:white;\n"
"}\n"
"QTabBar::tab{\n"
"height:35px;\n"
"background-color: #1d2643;\n"
"border-radius: 4px;\n"
"color:white;\n"
"}\n"
"QTabBar::tab:hover{\n"
"background-color: #1d2643;\n"
"color:white;\n"
"}\n"
"QTabBar::tab:selected{\n"
"background-color:white;\n"
"color: #1d2643;\n"
"}\n"
"#confidence::tab{\n"
"height:20px;\n"
"color:#1d2643;\n"
"}\n"
"#tabWidget{\n"
"background-color: #1d2643;\n"
"color:white;\n"
"font-size:20px;\n"
"}\n"
"#frame{\n"
"background-image: url(:/res/login_bg/detect.JPG);\n"
"}\n"
"#time_label{\n"
"border:none;\n"
"background-color:white;\n"
"}\n"
"#canl_label{\n"
"border:none;\n"
"background-color:white;\n"
"}\n"
"QPushButton{\n"
"height:35px;\n"
"background-color: #1d2643;\n"
"border-radius: 10px;\n"
"border: 2px groove gray;\n"
"border-style: outset;\n"
"color:white;\n"
"}\n"
"QLabel{\n"
"background-color: #1d2643;\n"
"border-radius: 100px;\n"
"border: 2px groove white;\n"
"}\n"
"QLineEdit{\n"
"font-size:20px;\n"
"border:none;\n"
"}\n"
"#pic{\n"
"    background-color:white;\n"
"}\n"
"#vid{\n"
"    border-image: url(:/res/login_bg/vid_cam.JPG);\n"
"}\n"
"#cam{\n"
"    border-image: url(:/res/login_bg/vid_cam.JPG);\n"
"}\n"
"#tabWidget_2{\n"
"background-color: #1d2643;\n"
"}\n"
"#btn_change_path{\n"
"border-image: url(:/res/24gf-folderOpen.png);\n"
"background:transparent;\n"
"}\n"
"#save_path_text{\n"
"border: 1px groove black;\n"
"}\n"
"#btn_right{\n"
"	border-image: url(:/res/\u7bad\u5934_\u5411\u53f3\u4e24\u6b21_o.png);\n"
"background:transparent;\n"
"}\n"
"#btn_left{\n"
"	border-image: url(:/res/\u7bad\u5934_\u5411\u5de6\u4e24\u6b21_o.png);\n"
"background:transparent;\n"
"}")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.btn_close_win = QtWidgets.QPushButton(self.frame)
        self.btn_close_win.setGeometry(QtCore.QRect(1370, 10, 31, 31))
        self.btn_close_win.setStyleSheet("border-image: url(:/res/关闭 (1).png);")
        self.btn_close_win.setObjectName("btn_close_win")
        self.btn_max_win = QtWidgets.QPushButton(self.frame)
        self.btn_max_win.setGeometry(QtCore.QRect(1320, 10, 31, 31))
        self.btn_max_win.setStyleSheet("border-image: url(:/res/Maximize-3.png);")
        self.btn_max_win.setObjectName("btn_max_win")
        self.btn_min_win = QtWidgets.QPushButton(self.frame)
        self.btn_min_win.setGeometry(QtCore.QRect(1270, 10, 31, 31))
        self.btn_min_win.setStyleSheet("border-image: url(:/res/最小化.png);")
        self.btn_min_win.setObjectName("btn_min_win")
        self.tabWidget = QtWidgets.QTabWidget(self.frame)
        self.tabWidget.setGeometry(QtCore.QRect(10, 60, 1390, 731))
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setIconSize(QtCore.QSize(16, 16))
        self.tabWidget.setObjectName("tabWidget")
        self.pic_det = QtWidgets.QWidget()
        self.pic_det.setObjectName("pic_det")
        self.pic = QtWidgets.QWidget(self.pic_det)
        self.pic.setGeometry(QtCore.QRect(0, 0, 1392, 711))
        self.pic.setObjectName("pic")
        self.btn_open_img = QtWidgets.QPushButton(self.pic)
        self.btn_open_img.setGeometry(QtCore.QRect(250, 640, 111, 35))
        self.btn_open_img.setObjectName("btn_open_img")
        self.btn_detect_img = QtWidgets.QPushButton(self.pic)
        self.btn_detect_img.setGeometry(QtCore.QRect(1000, 640, 111, 35))
        self.btn_detect_img.setObjectName("btn_detect_img")
        self.img_label = QtWidgets.QLabel(self.pic)
        self.img_label.setGeometry(QtCore.QRect(20, 20, 620, 620))
        self.img_label.setText("")
        self.img_label.setScaledContents(True)
        self.img_label.setObjectName("img_label")
        self.det_img_result = QtWidgets.QLabel(self.pic)
        self.det_img_result.setGeometry(QtCore.QRect(740, 20, 620, 620))
        self.det_img_result.setText("")
        self.det_img_result.setScaledContents(True)
        self.det_img_result.setObjectName("det_img_result")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/res/图片.png"), QtGui.QIcon.Disabled, QtGui.QIcon.On)
        icon.addPixmap(QtGui.QPixmap(":/res/图片.png"), QtGui.QIcon.Active, QtGui.QIcon.Off)
        icon.addPixmap(QtGui.QPixmap(":/res/login_bg/图片.png"), QtGui.QIcon.Active, QtGui.QIcon.On)
        icon.addPixmap(QtGui.QPixmap(":/res/图片.png"), QtGui.QIcon.Selected, QtGui.QIcon.Off)
        icon.addPixmap(QtGui.QPixmap(":/res/login_bg/图片.png"), QtGui.QIcon.Selected, QtGui.QIcon.On)
        self.tabWidget.addTab(self.pic_det, icon, "")
        self.vid_det = QtWidgets.QWidget()
        self.vid_det.setObjectName("vid_det")
        self.vid = QtWidgets.QWidget(self.vid_det)
        self.vid.setGeometry(QtCore.QRect(0, 0, 1391, 711))
        self.vid.setObjectName("vid")
        self.btn_close_video = QtWidgets.QPushButton(self.vid)
        self.btn_close_video.setGeometry(QtCore.QRect(870, 630, 160, 35))
        self.btn_close_video.setObjectName("btn_close_video")
        self.btn_open_video = QtWidgets.QPushButton(self.vid)
        self.btn_open_video.setGeometry(QtCore.QRect(580, 630, 160, 35))
        self.btn_open_video.setObjectName("btn_open_video")
        self.vid_label = QtWidgets.QLabel(self.vid)
        self.vid_label.setGeometry(QtCore.QRect(360, 10, 931, 611))
        self.vid_label.setText("")
        self.vid_label.setScaledContents(True)
        self.vid_label.setObjectName("vid_label")
        self.left_area_2 = QtWidgets.QGroupBox(self.vid)
        self.left_area_2.setGeometry(QtCore.QRect(20, 0, 321, 661))
        self.left_area_2.setStyleSheet("border:none;")
        self.left_area_2.setTitle("")
        self.left_area_2.setObjectName("left_area_2")
        self.label_5 = QtWidgets.QLabel(self.left_area_2)
        self.label_5.setGeometry(QtCore.QRect(10, 60, 301, 31))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("color:white;")
        self.label_5.setObjectName("label_5")
        self.label_7 = QtWidgets.QLabel(self.left_area_2)
        self.label_7.setGeometry(QtCore.QRect(10, 290, 301, 31))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_7.setFont(font)
        self.label_7.setStyleSheet("color:white;")
        self.label_7.setObjectName("label_7")
        self.result_label = QtWidgets.QLabel(self.left_area_2)
        self.result_label.setGeometry(QtCore.QRect(10, 97, 301, 181))
        self.result_label.setStyleSheet("background-color:white;\n"
                                   "font-size:25px;\n"
                                   "color:red;\n""text-align:center;")
        self.result_label.setText("")
        self.result_label.setObjectName("result_label")
        self.frame_video = QtWidgets.QFrame(self.left_area_2)
        self.frame_video.setGeometry(QtCore.QRect(10, 320, 301, 201))
        self.frame_video.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_video.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_video.setObjectName("frame_video")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/res/视频2.png"), QtGui.QIcon.Disabled, QtGui.QIcon.On)
        icon1.addPixmap(QtGui.QPixmap(":/res/视频2.png"), QtGui.QIcon.Active, QtGui.QIcon.Off)
        icon1.addPixmap(QtGui.QPixmap(":/res/login_bg/视频2.png"), QtGui.QIcon.Active, QtGui.QIcon.On)
        self.tabWidget.addTab(self.vid_det, icon1, "")
        self.cam_det = QtWidgets.QWidget()
        self.cam_det.setObjectName("cam_det")
        self.cam = QtWidgets.QWidget(self.cam_det)
        self.cam.setGeometry(QtCore.QRect(0, 0, 1391, 711))
        self.cam.setObjectName("cam")
        self.btn_open_camera = QtWidgets.QPushButton(self.cam)
        self.btn_open_camera.setGeometry(QtCore.QRect(730, 630, 180, 35))
        self.btn_open_camera.setObjectName("btn_open_camera")
        self.cam_label = QtWidgets.QLabel(self.cam)
        self.cam_label.setGeometry(QtCore.QRect(360, 10, 931, 611))
        self.cam_label.setText("")
        self.cam_label.setScaledContents(True)
        self.cam_label.setObjectName("cam_label")
        self.left_area = QtWidgets.QGroupBox(self.cam)
        self.left_area.setGeometry(QtCore.QRect(18, 7, 321, 661))
        self.left_area.setStyleSheet("border:none;")
        self.left_area.setTitle("")
        self.left_area.setObjectName("left_area")
        self.label = QtWidgets.QLabel(self.left_area)
        self.label.setGeometry(QtCore.QRect(12, 53, 301, 31))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label.setFont(font)
        self.label.setStyleSheet("color:white;")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.left_area)
        self.label_2.setGeometry(QtCore.QRect(12, 90, 301, 181))
        self.label_2.setStyleSheet("background-color:white;\n"
                                   "font-size:25px;\n"
                                   "color:red;\n""text-align:center;")
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.left_area)
        self.label_3.setGeometry(QtCore.QRect(12, 283, 301, 31))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("color:white;")
        self.label_3.setObjectName("label_3")
        self.frame_camera = QtWidgets.QFrame(self.left_area)
        self.frame_camera.setGeometry(QtCore.QRect(12, 313, 301, 201))
        self.frame_camera.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_camera.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_camera.setObjectName("frame_camera")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/res/w_摄像头.png"), QtGui.QIcon.Disabled, QtGui.QIcon.On)
        icon2.addPixmap(QtGui.QPixmap(":/res/w_摄像头.png"), QtGui.QIcon.Active, QtGui.QIcon.Off)
        icon2.addPixmap(QtGui.QPixmap(":/res/login_bg/w_摄像头.png"), QtGui.QIcon.Active, QtGui.QIcon.On)
        self.tabWidget.addTab(self.cam_det, icon2, "")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.listWidget = QtWidgets.QListWidget(self.tab)
        self.listWidget.setGeometry(QtCore.QRect(0, 10, 251, 681))
        self.listWidget.setAutoFillBackground(True)
        self.listWidget.setStyleSheet("background-color: rgb(29, 38, 67);")
        self.listWidget.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.listWidget.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.listWidget.setLineWidth(1)
        self.listWidget.setMidLineWidth(0)
        self.listWidget.setAutoScrollMargin(20)
        self.listWidget.setMovement(QtWidgets.QListView.Static)
        self.listWidget.setFlow(QtWidgets.QListView.TopToBottom)
        self.listWidget.setResizeMode(QtWidgets.QListView.Fixed)
        self.listWidget.setLayoutMode(QtWidgets.QListView.SinglePass)
        self.listWidget.setSpacing(30)
        self.listWidget.setGridSize(QSize(20, 80))
        self.listWidget.setUniformItemSizes(True)
        self.listWidget.setBatchSize(100)
        self.listWidget.setSelectionRectVisible(True)
        self.listWidget.setItemAlignment(QtCore.Qt.AlignBaseline|QtCore.Qt.AlignBottom|QtCore.Qt.AlignCenter|QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop|QtCore.Qt.AlignVCenter|QtCore.Qt.AlignVertical_Mask)
        self.listWidget.setObjectName("listWidget")
        item = QtWidgets.QListWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(50)
        font.setStyleStrategy(QtGui.QFont.NoAntialias)
        item.setFont(font)
        brush = QtGui.QBrush(QtGui.QColor(29, 38, 67))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item.setBackground(brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item.setForeground(brush)
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(50)
        # font.setStyleStrategy(QtGui.QFont.NoAntialias)
        item.setFont(font)
        brush = QtGui.QBrush(QtGui.QColor(29, 38, 67))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item.setBackground(brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        item.setForeground(brush)
        self.listWidget.addItem(item)
        self.stackedWidget = QtWidgets.QStackedWidget(self.tab)
        self.stackedWidget.setGeometry(QtCore.QRect(260, 10, 1121, 681))
        self.stackedWidget.setObjectName("stackedWidget")
        self.deploy = QtWidgets.QWidget()
        self.deploy.setObjectName("deploy")
        self.label_10 = QtWidgets.QLabel(self.deploy)
        self.label_10.setGeometry(QtCore.QRect(200, 40, 111, 31))
        self.label_10.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.deploy)
        self.label_11.setGeometry(QtCore.QRect(200, 100, 111, 31))
        self.label_11.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.save_path_text = QtWidgets.QLineEdit(self.deploy)
        self.save_path_text.setGeometry(QtCore.QRect(340, 40, 271, 31))
        self.save_path_text.setStyleSheet("")
        self.save_path_text.setObjectName("save_path_text")

        self.det_device_text = QtWidgets.QLineEdit(self.deploy)
        self.det_device_text.setGeometry(QtCore.QRect(340, 100, 271, 31))
        self.det_device_text.setStyleSheet("border: 1px groove black;")
        self.det_device_text.setObjectName("warn_info_text")

        # self.env_combo = QtWidgets.QComboBox(self.deploy)
        # self.env_combo.setGeometry(QtCore.QRect(340, 100, 271, 31))
        # font = QtGui.QFont()
        # font.setPointSize(13)
        # font.setBold(True)
        # font.setWeight(75)
        # self.env_combo.setFont(font)
        # self.env_combo.setObjectName("env_combo")
        # self.env_combo.addItem("")
        # self.env_combo.addItem("")

        self.btn_change_path = QtWidgets.QPushButton(self.deploy)
        self.btn_change_path.setGeometry(QtCore.QRect(580, 40, 31, 31))
        self.btn_change_path.setAutoFillBackground(False)
        self.btn_change_path.setStyleSheet("")
        self.btn_change_path.setText("")
        self.btn_change_path.setObjectName("btn_change_path")

        self.stackedWidget.addWidget(self.deploy)
        self.statistic = QtWidgets.QWidget()
        self.statistic.setObjectName("statistic")
        self.label_4 = QtWidgets.QLabel(self.statistic)
        self.label_4.setGeometry(QtCore.QRect(200, 40, 111, 31))
        self.label_4.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.warn_time_combo = QtWidgets.QComboBox(self.statistic)
        self.warn_time_combo.setGeometry(QtCore.QRect(330, 40, 201, 31))
        self.warn_time_combo.setStyleSheet("border: 1px groove black;")
        self.warn_time_combo.setObjectName("warn_time_combo")
        self.warn_time_combo.addItem("")
        self.warn_time_combo.addItem("")
        self.warn_time_combo.addItem("")
        self.label_6 = QtWidgets.QLabel(self.statistic)
        self.label_6.setGeometry(QtCore.QRect(200, 100, 111, 31))
        self.label_6.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.label_8 = QtWidgets.QLabel(self.statistic)
        self.label_8.setGeometry(QtCore.QRect(200, 160, 111, 31))
        self.label_8.setStyleSheet("color: rgb(255, 255, 255);")
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.warn_info_text = QtWidgets.QLineEdit(self.statistic)
        self.warn_info_text.setGeometry(QtCore.QRect(330, 100, 601, 31))
        self.warn_info_text.setStyleSheet("border: 1px groove black;")
        self.warn_info_text.setObjectName("warn_info_text")
        self.warn_pic = QtWidgets.QLabel(self.statistic)
        self.warn_pic.setGeometry(QtCore.QRect(330, 160, 601, 511))
        self.warn_pic.setScaledContents(True)
        self.warn_pic.setAlignment(QtCore.Qt.AlignCenter)
        self.warn_pic.setObjectName("warn_pic")
        self.btn_left = QtWidgets.QPushButton(self.statistic)
        self.btn_left.setObjectName(u"btn_left")
        self.btn_left.setGeometry(QtCore.QRect(264, 360, 51, 91))
        self.btn_right = QtWidgets.QPushButton(self.statistic)
        self.btn_right.setObjectName(u"btn_right")
        self.btn_right.setGeometry(QtCore.QRect(950, 360, 51, 91))
        self.stackedWidget.addWidget(self.statistic)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/res/设置 (3).png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon3.addPixmap(QtGui.QPixmap(":/res/设置 (2).png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.tabWidget.addTab(self.tab, icon3, "")
        self.canl_label = QtWidgets.QLabel(self.frame)
        self.canl_label.setGeometry(QtCore.QRect(1000, 60, 31, 31))
        self.canl_label.setText("")
        self.canl_label.setPixmap(QtGui.QPixmap(":/res/login_bg/24gf-calendar.png"))
        self.canl_label.setScaledContents(True)
        self.canl_label.setObjectName("canl_label")
        self.date_text = QtWidgets.QLineEdit(self.frame)
        self.date_text.setGeometry(QtCore.QRect(1040, 60, 191, 31))
        self.date_text.setObjectName("date_text")
        self.time_label = QtWidgets.QLabel(self.frame)
        self.time_label.setGeometry(QtCore.QRect(1240, 61, 31, 31))
        self.time_label.setText("")
        self.time_label.setPixmap(QtGui.QPixmap(":/res/login_bg/时间.png"))
        self.time_label.setScaledContents(True)
        self.time_label.setObjectName("time_label")
        self.time_text = QtWidgets.QLineEdit(self.frame)
        self.time_text.setGeometry(QtCore.QRect(1280, 60, 121, 31))
        self.time_text.setObjectName("time_text")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(3)
        self.listWidget.setCurrentRow(-1)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

        MainWindow.setWindowFlags(Qt.FramelessWindowHint)  # 隐藏标题栏
        MainWindow.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # 将窗体背景透明

        self.btn_open_img.setText(_translate("MainWindow", "导入图片"))
        self.btn_detect_img.setText(_translate("MainWindow", "检测图片"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.pic_det), _translate("MainWindow", "图片检测"))
        self.btn_close_video.setText(_translate("MainWindow", "停止检测"))
        self.btn_open_video.setText(_translate("MainWindow", "导入视频实时监测"))
        self.label_5.setText(_translate("MainWindow", "| 检测结果"))
        self.label_7.setText(_translate("MainWindow", "| 检测结果置信度分布"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.vid_det), _translate("MainWindow", "视频检测"))
        self.btn_open_camera.setText(_translate("MainWindow", "打开摄像头实时监测"))
        self.label.setText(_translate("MainWindow", "| 检测结果"))
        self.label_3.setText(_translate("MainWindow", "| 检测结果置信度分布"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.cam_det), _translate("MainWindow", "摄像头检测"))

        self.listWidget.setSortingEnabled(False)
        __sortingEnabled = self.listWidget.isSortingEnabled()
        self.listWidget.setSortingEnabled(False)
        item = self.listWidget.item(0)
        item.setText(_translate("MainWindow", "相关配置"))
        self.listWidget.setSortingEnabled(__sortingEnabled)
        item = self.listWidget.item(1)
        item.setText(_translate("MainWindow", "查看统计"))
        self.listWidget.setSortingEnabled(__sortingEnabled)
        self.label_10.setText(_translate("MainWindow", "更改存图路径"))
        self.label_11.setText(_translate("MainWindow", "查看运行配置"))

        # self.env_combo.setCurrentText(_translate("MainWindow", "cpu"))
        # self.env_combo.setItemText(0, _translate("MainWindow", "cpu"))
        # self.env_combo.setItemText(1, _translate("MainWindow", "0"))

        self.label_4.setText(_translate("MainWindow", "信息来源："))
        self.warn_time_combo.setCurrentText(_translate("MainWindow", "摄像头"))
        self.warn_time_combo.setItemText(0, _translate("MainWindow", "摄像头"))
        self.warn_time_combo.setItemText(1, _translate("MainWindow", "视频"))
        self.warn_time_combo.setItemText(2, _translate("MainWindow", "图片"))
        self.label_6.setText(_translate("MainWindow", "报警信息："))
        self.label_8.setText(_translate("MainWindow", "对应图片："))
        self.warn_pic.setText(_translate("MainWindow", "图片"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "系统设置"))
        self.date_text.setText(_translate("MainWindow", "2020-04-20 星期五"))
        self.time_text.setText(_translate("MainWindow", "22:35:50"))

        self.btn_min_win.clicked.connect(self.right_mini_click)
        self.btn_max_win.clicked.connect(self.right_maxi_click)
    #     self.listWidget.itemClicked.connect(self.item_clicked)
    #
    # def item_clicked(self):
    #     item = self.listWidget.currentItem().text()
    #     print('item', item)
    #     if item == '更改配置':
    #         self.stackedWidget.setCurrentIndex(0)
    #     elif item == '查看统计':
    #         self.stackedWidget.setCurrentIndex(1)

    def right_mini_click(self):  # 最小化按钮单击事件
        self.showMinimized()

    def right_maxi_click(self):  # 最大化按钮单击事件
        pass

    def mousePressEvent(self, event):  # 鼠标拖拽窗体
        if event.button() == Qt.LeftButton:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QCursor(Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseMoveEvent(self, QMouseEvent):  # 鼠标拖拽窗体
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)  # 更改窗口位置
            QMouseEvent.accept()

    def mouseReleaseEvent(self, QMouseEvent):  # 鼠标拖拽窗体
        self.m_flag = False
        self.setCursor(QCursor(Qt.ArrowCursor))

import res.img_rc
import sys


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QMainWindow()  # #####
    ui = Ui_MainWindow()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

