# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect2.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from utils.augmentations import letterbox
import warnings

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.general import check_img_size, check_suffix, non_max_suppression, print_args, \
    scale_coords, set_logging
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync

# Initialize
# device = select_device('cpu')

# Load model


warnings.filterwarnings('ignore')
stride, names, model, imgsz = None, None, None, None


class detectImg:
    def __init__(self):
        self.opt = self.parse_opt()
        self.device = select_device(device=self.opt.device)
        print("device:", self.opt.device)
        print("weight:", self.opt.weights)
        self.initialize_model(self.opt.weights)

    def initialize_model(self, weight):
        global stride, names, model, imgsz
        stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
        model = attempt_load(weight, map_location=self.device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names 所有类别{001 002 004}

        imgsz = check_img_size(640, s=stride)  # check image size

    @torch.no_grad()
    def run(self, image, weights=ROOT / 'best.pt',  # model.pt path(s)
            source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
            imgsz=640,  # inference size (pixels)
            conf_thres=0.2,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=ROOT / 'runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            dnn=False,  # use OpenCV DNN for ONNX inference
            ):
        source = str(source)
        # save_img = not nosave and not source.endswith('.txt')  # save inference images
        #
        # # Directories
        # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Dataloader

        # dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        # bs = 1  # batch_size
        # vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model(torch.zeros(1, 3, *imgsz).to(self.device).type_as(next(model.parameters())))  # run once
        dt, seen = [0.0, 0.0, 0.0], 0
        # for path, img, im0s, vid_cap in dataset:
        t1 = time_sync()
        im0s = image
        img = letterbox(im0s, imgsz, stride=stride, auto=True)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference

        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(img, augment=augment, visualize=visualize)[0]
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Process predictions预测过程
        for i, det in enumerate(pred):  # per image
            seen += 1
            s, im0, = '', im0s.copy()

            # p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    #hide_conf = True
                    #hide_labels = True
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Print time (inference-only)
            # print(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            #
            # # Save results (image with detections)
            # if save_img:
            #     cv2.imwrite(save_path, im0)

        return im0, pred, names

    def parse_opt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weight/best-s.pt', help='model path(s)')  # ########权重
        parser.add_argument('--source', type=str, default=ROOT / 'images', help='file/dir/URL/glob, 0 for webcam')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.23, help='confidence threshold')    # ####################置信度
        parser.add_argument('--iou-thres', type=float, default=0.40, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')  # ###############
        parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')  # ##############设备
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_false', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')  # #######框的像素
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')  # ##################隐藏标签 ture为隐藏
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')  # ################隐藏置信度
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
        self.opt = parser.parse_args()
        self.opt.imgsz *= 2 if len(self.opt.imgsz) == 1 else 1  # expand
        # print_args(FILE.stem, opt)
        return self.opt

    # self.opt = parse_opt()

    # 主执行函数
    def predictImage(self, img):
        print("准备检测")
        outImg, pred, names = self.run(image=img, **vars(self.opt))
        print(pred)
        new_list = []    # 这个列表包含所有检测的结果的标签
        nomask_rate_list = []  # 这个列表包含检测到不带口罩的置信度
        mask_rate_list = []  # 这个列表包含检测到带口罩的置信度
        list_pred = pred[0].tolist()
        for li in list_pred:
            new_list.append(int(li[-1]))
            if int(li[-1]) == 0:
                nomask_rate_list.append(li[-2])
            else:
                mask_rate_list.append(li[-2])
        print("new_list", new_list)   # 类列表
        # 列表去重
        format_list = list(set(new_list))
        format_list.sort(key=new_list.index)
        print("format_list", format_list)  # 类列表
        if len(pred[0]) == 0:
            info = 0
            return outImg, info, format_list, nomask_rate_list, mask_rate_list, new_list
        else:
            info = 1
            # info2 = len(pred[0])
            return outImg, info, format_list, nomask_rate_list, mask_rate_list, new_list
            # 如果要记录类型 在此操作
# 把图放进去 outImg是打框以后的图 pred是框的坐标值

if __name__ == "__main__":
    label_class_dict = {1: "mask", 0: "no mask"}
    test1 = detectImg()
    test1.initialize_model('weight/best-s.pt')
    img = cv2.imread('./images/1-1.jpg')  # 读图
    outImg, info, format_list, nomask_rate_list, mask_rate_list, new_list = test1.predictImage(img)
    # 将置信度列表的每一个浮点数转换成保留两位有效数字
    mask_np = np.array(mask_rate_list)
    mask_np_2f = np.round(mask_np, 2)
    mask_new_list = list(mask_np_2f)
    # print("mask_new_list", mask_new_list)
    nomask_np = np.array(nomask_rate_list)
    nomask_np_2f = np.round(nomask_np, 2)
    nomask_new_list = list(nomask_np_2f)
    print('nomask_rate_list:  ', nomask_rate_list)
    print('mask_rate_list:  ', mask_rate_list)
    for li in format_list:
        print('++++', label_class_dict[li])
    # cv2.imwrite('8.jpg', outImg)  # 显示出来看看
    cv2.imshow('show', outImg)
    cv2.waitKey(0)  # 按键退出
