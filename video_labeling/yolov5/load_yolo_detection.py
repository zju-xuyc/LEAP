import os
import torch
from yolov5.models.common import DetectMultiBackend

def load_yolo_detection(model_name,
                        input_size=640,
                        data='yolov5/data/coco128.yaml',
                        dnn=False, half=True,
                        device="cpu"):
    model = DetectMultiBackend(model_name, device=torch.device("cuda"), data=data, fp16=half)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    if isinstance(input_size, int):
        model.warmup(imgsz=(1, 3, input_size, input_size))
    else:
        model.warmup(imgsz=(1, 3, input_size[0], input_size[1]))
    return model, stride, pt