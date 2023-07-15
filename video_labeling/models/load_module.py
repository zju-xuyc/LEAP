import os
import torch
# from defaults import _C as cfg
from .common import DetectMultiBackend
from .common_emb import Backbone
# from yolov4_models import *

model_info = {
    "yolov5x": ["yolov5x.pt"]
}

def load_yolo_detection(model_name,
                        input_size=640,
                        data_name="coco.yaml",
                        dnn=False, half=True,
                        device="cpu"):
    # if "yolov5" in model_name:
    model = DetectMultiBackend("./video_labeling/yolov5x.pt", device=torch.device("cuda"))
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    model.warmup(imgsz=(1, 3, input_size, input_size), half=half)
    return model, stride, pt

def load_embedding_detection(cfg):
    model = Backbone(cfg, num_classes=10).cuda()
    model.load_param(cfg.MODEL.EMB_WEIGHT)
    model.eval()
    model.warmup(torch.rand([1, 3, 384, 384]))

    return model