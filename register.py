# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'register.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt


class Ui_RegisterMain(object):
    def setupUi(self, RegisterMain):
        RegisterMain.setObjectName("RegisterMain")
        RegisterMain.resize(1187, 682)
        RegisterMain.setMaximumSize(QtCore.QSize(1187, 682))
        RegisterMain.setStyleSheet("QLineEdit{\n"
"border:none;\n"
"}\n"
"QPushButton{\n"
"background-color: #ffffff;\n"
"border-radius: 10px;\n"
"border: 2px groove gray;\n"
"border-style: outset;\n"
"}")
        self.centralwidget = QtWidgets.QWidget(RegisterMain)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, -10, 1631, 831))
        self.label.setStyleSheet("background-image: url(:/res/register.JPG);")
        self.label.setText("")
        self.label.setObjectName("label")
        self.btn_register = QtWidgets.QPushButton(self.centralwidget)
        self.btn_register.setGeometry(QtCore.QRect(480, 400, 111, 31))
        self.btn_register.setObjectName("btn_register")
        self.btn_quit = QtWidgets.QPushButton(self.centralwidget)
        self.btn_quit.setGeometry(QtCore.QRect(690, 400, 111, 31))
        self.btn_quit.setObjectName("btn_quit")
        self.user_name = QtWidgets.QLineEdit(self.centralwidget)
        self.user_name.setGeometry(QtCore.QRect(480, 232, 311, 40))
        self.user_name.setObjectName("lineEdit")
        self.user_password = QtWidgets.QLineEdit(self.centralwidget)
        self.user_password.setGeometry(QtCore.QRect(480, 300, 311, 41))
        self.user_password.setObjectName("lineEdit_2")
        # RegisterMain.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(RegisterMain)
        self.statusbar.setObjectName("statusbar")
        # RegisterMain.setStatusBar(self.statusbar)

        self.retranslateUi(RegisterMain)
        QtCore.QMetaObject.connectSlotsByName(RegisterMain)

    def retranslateUi(self, RegisterMain):
        _translate = QtCore.QCoreApplication.translate
        RegisterMain.setWindowTitle(_translate("RegisterMain", "LoginMain"))
        RegisterMain.setWindowFlags(Qt.FramelessWindowHint)  # ?????????????????????
        RegisterMain.setAttribute(QtCore.Qt.WA_TranslucentBackground)  # ????????????????????????
        self.btn_register.setText(_translate("RegisterMain", "??????"))
        self.btn_quit.setText(_translate("RegisterMain", "??????"))
import res.img_rc
