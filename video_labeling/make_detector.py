import torch

from configs import settings as settings
import numpy as np
import time
from configs import cfg

from video_labeling.yolov5.load_yolo_detection import load_yolo_detection
from video_labeling.yolov5.utils.augmentations import letterbox_first, letterbox_second
from video_labeling.yolov5.utils.general import non_max_suppression, scale_boxes

class Detector():  # ByteTracker
   def __init__(self, model_name, cfg, augment=False, visualize=False):
        self.model, self.stride, self.pt = load_yolo_detection(model_name,
                                                               input_size=cfg.INPUT.SIZE)
        self.model.eval()
        self.detactor_time = []
        self.augment = augment
        self.visualize = visualize
        
   def preprocess(self, cfg, size):
        # self.init_img = cv2.imread(init_img)
        self.init_img = np.random.rand(size[0], size[1], 3)
        _, *self.pad_args = letterbox_first(np.zeros_like(self.init_img), cfg.INPUT.SIZE, stride=self.stride, auto=self.pt)

   def __call__(self, impath, timer):
        # self.img0 = cv2.imread(impath)
        self.img0 = impath
        img = letterbox_second(self.img0, *self.pad_args)
        img = (torch.HalfTensor(np.ascontiguousarray(img.transpose((2, 0, 1))[::-1]))/255.0).cuda().unsqueeze(0)
        timer.tic()
        t1 = time.time()
        with torch.no_grad():
            if self.pt:
                pred = self.model(img, augment=False, visualize=False)[0]
            else:
                pred = self.model(img, augment=False, visualize=False)[-1]
        
        pred = non_max_suppression(pred, 
                                   cfg.MODEL.DETECT_CONF_THRE,
                                   cfg.MODEL.DETECT_IOU_THRE,
                                   cfg.MODEL.DETECT_CLASSES,
                                   cfg.MODEL.DETECT_AGNOSTIC_NMS, 
                                   max_det=cfg.MODEL.DETECT_MAX_DETECT)
        det = pred[0]
        
        det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], self.img0.shape).round()
        
        detect_infos = [None for _ in range(len(det))]
        det_idx = 0
        for *xyxy, conf, class_obj in reversed(det):
            detect_infos[det_idx] = torch.Tensor([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), conf, class_obj])
            det_idx += 1
        if len(detect_infos) == 0:
            t2 = time.time()
            self.detactor_time.append(t2 - t1)
            return [], False
        detect_infos = torch.stack(detect_infos, dim=0)
        t2 = time.time()
        self.detactor_time.append(t2 - t1)
        return detect_infos, True