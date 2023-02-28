
# from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
from login import Ui_LoginMain
from my_register import Register_Main
from sql_db import Db
from myprocess import MainMain
from PyQt5 import QtWidgets


class Login_Main(QWidget, Ui_LoginMain):
    def __init__(self):
        super(Login_Main, self).__init__()
        self.setupUi(self)
        self.initUi()
        self.mysql = Db()
        self.main = MainMain()
        self.my_register = Register_Main()
        self.data = []

    def initUi(self):
        self.btn_login.clicked.connect(self.login_clicked)
        self.btn_quit.clicked.connect(self.quit_clicked)
        self.btn_href.clicked.connect(self.href_register)

    def login_clicked(self):
        print("点击登录按钮")
        account = self.username.text()  # 获取输入的用户名
        print(len(account))
        password = self.password.text()  # 获取输入的密码
        print("1111111")
        if len(account) == 0 or len(password) == 0:
            QMessageBox.warning(self, "警告", "用户名或密码不能为空！")
        else:
            self.data = self.mysql.get_login_data(account)   # 将输入的用户名与数据库匹配
            print(len(self.data))
            if len(self.data) == 0:
                QtWidgets.QMessageBox.critical(self, "提示", "该用户不存在，请重新输入！")    # 返回的是一个空列表
            else:
                if self.data[0][1] == password:
                    self.main.show()  # 打开检测主窗体
                    self.close()  # 关闭登录窗体
                else:
                    QtWidgets.QMessageBox.critical(self, "提示", "密码错误，请重新输入！")

    # 点击退出按钮
    def quit_clicked(self):
        self.close()

    # 点击注册按钮
    def href_register(self):
        self.my_register.show()
        # self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Login_Main()
    w.show()
    sys.exit(app.exec_())