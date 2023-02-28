from datetime import datetime
from sql_db import Db
import os
import cv2


class ImageSave:
    def __init__(self):
        self.path = 'G:/data'
        self.data_path = None
        self.db = Db()

    # 图片存储地址
    def set_path(self, path):
        self.path = path

    # 返回图片路径
    def img_path(self):
        return self.pic_path

    # 创建文件夹，如果存在则不用创建
    def creat_dir(self):
        self.data_path = os.path.join(self.path, 'detect_dir')  # #######
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

    # 用于记录no mask带框的图片 info：no mask的
    def save_detect_img(self, img, pic_source):
        self.creat_dir()
        now = datetime.now()
        print('now', now)
        time_string = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
        print('time_string', time_string)
        pic_path = self.data_path + "/" + "{0}.jpg".format(time_string)
        print("pic_path", pic_path)
        cv2.imwrite(pic_path, img)
        if pic_source == '0':
            self.db.insert_cam_warn_info(pic_path)
        elif pic_source == '1':
            self.db.insert_vid_warn_info(pic_path)


# if __name__ == "__main__":
#     test = ImageSave()
#     test.setsrc('G:\data')
#     # test.creat_dir()
#     path = r'./images/qy.jpg'
#     # path = r'G:\data\detect_dir/22_10_31_881334.jpg'
#     img = cv2.imread(path)
#     test.save_detect_img(img)