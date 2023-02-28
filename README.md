# mask_dectect
 
一、安装anaconda后
1、打开anaconda
2、找到 import 按钮，点击会弹出一个弹窗
 
3、点数字1 找到mask_env.yml 文件，数字2处填写环境名
4、点击 import 按钮，即可
5、等待自动安装环境
# 三、系统运行操作说明
在pycharm打开 mask_sys 系统程序，点击运行 my_login.py 文件，弹出登录窗体，管理员账号：admin，密码：123
注意：本系统是用sqlserver 2012 数据库，数据库脚本文件为maskdb.sql。另外若需要进行视频检测则需要准备带有摄像头的电脑。
为了使用方便，系统也使用了sqlite数据库，该版本是用的sqlite,若需要使用SqlServer则需要到sql_db.py 文件中修改连接字符串。
系统默认运行环境为CPU，若在gpu 上运行则需要修改detect.py 文件的 device 属性。
