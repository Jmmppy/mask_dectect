import sys
from register import Ui_RegisterMain
from PyQt5.QtWidgets import *
from sql_db import Db


class Register_Main(QWidget, Ui_RegisterMain):
    def __init__(self):
        super(Register_Main, self).__init__()
        self.setupUi(self)
        self.slot_initUi()
        self.db = Db()

    def slot_initUi(self):
        self.btn_register.clicked.connect(self.register)
        self.btn_quit.clicked.connect(self.quit_register)

    def register(self):
        account = self.user_name.text()
        print('===============', account)
        password = self.user_password.text()
        print('===============', password)
        a = self.db.judge_account(account)
        print('-----------------a', a)
        if a == 1:
            self.user_name.clear()
            self.user_password.clear()
            QMessageBox.warning(self, "警告", "{0}用户名已存在!".format(account))
        elif a == 0:
            print('-----------------a', a)
            self.db.add_account(account, password)
            print('-----------------a', a)
            self.user_name.clear()
            self.user_password.clear()

    def quit_register(self):
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Register_Main()
    w.show()
    sys.exit(app.exec_())