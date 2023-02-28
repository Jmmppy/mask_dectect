# import pyodbc
import sqlite3
from datetime import datetime
# from pyodbc import Error


class Db:
    def __init__(self):
        self.db_file = 'maskdb.db'
        pass

    @staticmethod
    def get_db_conn():
        conn = None
        conn = sqlite3.connect('maskdb.db')  # sqlite 数据库
        # try:
        #     conn = sqlite3.connect('maskdb.db')     # sqlite 数据库
        #     # conn = pyodbc.connect(r'DRIVER={SQL Server Native Client 11.0};'r'SERVER=LJY\MSSQLSERVER2012;'r'DATABASE=maskdb;'r'UID=sa;'r'PWD=123')
        # except Error as e:
        #     print(e)
        if conn is not None:
            return conn

    @staticmethod
    def close_db_conn(cur, conn):
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    def get_data(self):
        conn = self.get_db_conn()
        cur = conn.cursor()
        a = cur.execute("select * from users")
        rows = cur.fetchall()
        print(rows)
        self.close_db_conn(cur, conn)

    def get_login_data(self, account):
        print("准备读取数据")
        conn = self.get_db_conn()
        cur = conn.cursor()
        sql = "select useName, passWord from users where useName=?"
        result = cur.execute(sql, (account,))
        data = result.fetchall()
        self.close_db_conn(cur, conn)
        return data

    def common(self, conn, sql):  # 将数据从sql格式转换成列表镶嵌字典的格式并返回
        cursor = conn.execute(sql)
        information = []
        for row in cursor:
            information.append(row)
        return information

    def judge_account(self, account):   # 用于判断账户是否存在
        conn = self.get_db_conn()
        cur = conn.cursor()
        sql = "select distinct useName from users"
        accounts = self.common(conn, sql)
        self.close_db_conn(cur, conn)
        print(accounts)
        for acc in accounts:
            if acc[0] == account:
                return 1  # 返回1说明账号存在
        return 0

    # 注册用户
    def add_account(self, account, password):  # 增加记录
        conn = self.get_db_conn()
        cur = conn.cursor()
        now = datetime.now()
        create_time = now.strftime("%Y_%m_%d")
        sql = "insert into users (useName, passWord, createTime) values ('%s','%s','%s')" % (account, password, create_time)
        conn.execute(sql)
        conn.commit()
        self.close_db_conn(cur, conn)

    # 插入摄像头警报信息及图片路径、信息来源
    def insert_cam_warn_info(self, pic_path):
        conn = self.get_db_conn()
        cur = conn.cursor()
        now = datetime.now()
        create_time = now.strftime("%Y_%m_%d")
        # creat_time = time.time()
        print("type(creat_time)", type(create_time))
        source = '摄像头'
        sql = "insert into warning_info (pic_path, warn_time, info_source) values ('%s','%s','%s')" % (pic_path, create_time, source)
        cur.execute(sql)
        conn.commit()
        self.close_db_conn(cur, conn)

    # 插入视频警报信息及图片路径、信息来源
    def insert_vid_warn_info(self, pic_path):
        conn = self.get_db_conn()
        cur = conn.cursor()
        now = datetime.now()
        creat_time = now.strftime("%Y_%m_%d")
        source = '视频'
        sql = "insert into warning_info (pic_path, warn_time, info_source) values (?,?,?)"
        data = (pic_path, creat_time, source)
        cur.execute(sql, data)
        conn.commit()
        self.close_db_conn(cur, conn)

    # 根据信息来源从数据库读取对应的图片地址到列表中
    def select_pic_info(self, info_source):
        conn = self.get_db_conn()
        cur = conn.cursor()
        sql = "select pic_path, warn_time from warning_info where info_source =?"
        result = cur.execute(sql, (info_source,))
        pic_path_list = []
        data = result.fetchall()
        print('data------------', data)
        for row in data:
            pic_path_list.append(row[0])    # 读取每一个元组的第一项
        print("pic_path_list", pic_path_list)   # 图片地址列表
        pic_list_len = len(pic_path_list)
        print(len(pic_path_list))
        self.close_db_conn(cur, conn)
        return pic_path_list

    # 获取图片对应的时间
    def pic_warn_time(self, path):
        conn = self.get_db_conn()
        cur = conn.cursor()
        sql = "select warn_time from warning_info where pic_path =?"
        result = cur.execute(sql, (path,))
        data = result.fetchall()
        self.close_db_conn(cur, conn)
        for acc in data:
            data = acc[0]
        return data


if __name__ == "__main__":
    test = Db()
    a = test.judge_account('jying')
    print(a)
    # test.add_account('test', '123')
    test.get_data()
    a = test.judge_account('admin')
    print(a)
    path = 'G:/data\\detect_dir/2022_06_01_16_54_13_406291.jpg'
    b = test.pic_warn_time(path)
    print(b)
    # path ='G:\data\detect_dir/22_10_31_881334.jpg'
    # test.insert_vid_warn_info(path)
    # source = '视频'
    # pic_path_list = test.select_pic_info(source)
    # for i in range(len(pic_path_list)):
    #     print(i)
    #     a = pic_path_list[i]
    #     print(a)

