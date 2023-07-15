import cv2
import torch
import numpy as np
from detect_utils.datasets import letterbox
from models.experimental import attempt_load
from detect_utils.general import check_img_size, non_max_suppression, scale_coords
from detect_utils.torch_utils import select_device
import time
import warnings
from video_labeling.models import load_yolo_detection
from tools.utils import letterbox_first, letterbox_second, non_max_suppression, scale_coords
from configs import cfg

warnings.filterwarnings("ignore", category=UserWarning)

class Detector(object):

    def __init__(
        self, model_path, input_size,
        device='0', conf_thres=0.5, iou_thres=0.5
    ):

        self.weights = model_path
        self.imgsz = input_size
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'
        self.model = None
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self._init_model()

    def _init_model(self):
        self.model = attempt_load(self.weights, map_location=self.device)
        self.model.eval()
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(
            self.imgsz, s=self.stride)  # check img_size
        self.names = self.model.module.names if hasattr(
            self.model, 'module') else self.model.names  # get class names
        if self.half:
            self.model.half()  # to FP16

    def preprocess(self, img0):
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect(self, img0):

        img = self.preprocess(img0)
        pred = self.model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, classes=None, agnostic=False)

        bboxes = []
        for i, det in enumerate(pred):  # detections per image

            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to img0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], img0.shape).round()

                for value in reversed(det):
                    xyxy, conf, cls_id = value[:4], value[4], value[5]
                    if int(cls_id) in [2,5,7]:
                        lbl = self.names[int(cls_id)]
                        x1, y1 = int(xyxy[0]), int(xyxy[1])
                        x2, y2 = int(xyxy[2]), int(xyxy[3])
                        line = [x1, y1, x2, y2, float(conf), int(cls_id)]
                        bboxes.append(line)
        return img0, bboxes
    
class Detector_origin():  # ByteTracker
    
   def __init__(self, cfg):

        self.model, self.stride, self.pt = load_yolo_detection(cfg)
        self.model.eval()
        self.detactor_time = []

   def preprocess(self, cfg, size):
        # self.init_img = cv2.imread(init_img)
        self.init_img = np.random.rand(size[0], size[1], 3)
        _, *self.pad_args = letterbox_first(np.zeros_like(self.init_img), cfg.INPUT.SIZE, stride=self.stride, auto=self.pt)

   def __call__(self, impath):
        # self.img0 = cv2.imread(impath)
        self.img0 = impath
        img = letterbox_second(self.img0, *self.pad_args)
        img = (torch.HalfTensor(np.ascontiguousarray(img.transpose((2, 0, 1))[::-1]))/255.0).cuda().unsqueeze(0)
        t1 = time.time()
        with torch.no_grad():
            pred = self.model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, cfg.MODEL.DETECT_CONF_THRE, cfg.MODEL.DETECT_IOU_THRE, cfg.MODEL.DETECT_CLASSES, cfg.MODEL.DETECT_AGNOSTIC_NMS, max_det=cfg.MODEL.DETECT_MAX_DETECT)
        det = pred[0]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], self.img0.shape).round()
        detect_infos = [None for _ in range(len(det))]
        det_idx = 0
        for *xyxy, conf, class_obj in reversed(det):
            detect_infos[det_idx] = torch.Tensor([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), conf, class_obj])
            det_idx += 1
        if len(detect_infos) == 0:
            return [], False
        detect_infos = torch.stack(detect_infos, dim=0)
        t2 = time.time()
        self.detactor_time.append(t2 - t1)
        return detect_infos, True

if __name__ == '__main__':

    img0 = cv2.imread('XXX')
    det = Detector(model_path='./weights/yolov5l.pt',
                   input_size=640, conf_thres=0.2)
    result, bboxes = det.detect(img0.copy())
    # cv2.imshow('result', result)
    for x1, y1, x2, y2, lbl in bboxes:
        cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0))
        cv2.putText(img0, lbl, (x1, y1),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.imwrite("tmp.jpg", img0)





