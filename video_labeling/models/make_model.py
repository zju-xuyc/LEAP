from copyreg import pickle
from dis import dis
import os
import cv2
import copy
import numpy as np
import torch
import pickle
import torch.nn as nn
from scipy.spatial.distance import cdist
from PIL import Image, ImageDraw, ImageFont
from .load_module import load_yolo_detection, load_embedding_detection
from .utils import non_max_suppression, scale_coords, colors

class Baseline(nn.Module):
    def __init__(self, cfg):
        super(Baseline, self).__init__()
        self.detect, self.stride, self.pt = load_yolo_detection(cfg)
        self.embedding = load_embedding_detection(cfg)
        self.conf_thres = cfg.MODEL.DETECT_CONF_THRE
        self.iou_thres = cfg.MODEL.DETECT_IOU_THRE
        self.classes = cfg.MODEL.DETECT_CLASSES
        self.agnostic_nms = cfg.MODEL.DETECT_AGNOSTIC_NMS
        self.max_det = cfg.MODEL.DETECT_MAX_DETECT
        self.visual_dir = cfg.OUTPUT.VISUAL_DETECT
        self.embed_dir = cfg.OUTPUT.EMBED_DETECT
        self.embed_step = cfg.TEST.EMBED_STEP
        self.gt_thre = cfg.DATASET.GT_THRE
        self.class_names = self.detect.names
        self.all_cares = []
        self.transform = None
        self.test_gt_bbox_dict = None
        self.lw = 3 or max(round(640 * 0.003), 2)
        self.detect_imgs = []
        self.detect_infos = []
        self.vids = []

    def init_transform(self, transform, test_gt_bbox_dict):
        self.transform = transform
        self.test_gt_bbox_dict = test_gt_bbox_dict

    def extract_embed(self, x):
        feat = self.embedding(x).cpu().numpy()
        return feat

    def process_pred(self, p, s, rs, v, f, m):
        p = non_max_suppression(p, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
        for batch_i, (det, shape, video, frame, image_raw) in enumerate(zip(p, rs, v, f, m)):
            if len(self.detect_infos) > self.embed_step:
                detect_imgs = torch.stack(self.detect_imgs, dim=0).cuda()
                feat = self.embedding(detect_imgs).cpu().numpy()
                for i in range(len(self.detect_infos)):
                    v, fra, did, p_p = self.detect_infos[i]
                    f = feat[i]
                    if not os.path.exists(os.path.join(self.embed_dir, video)):
                        os.mkdir(os.path.join(self.embed_dir, video))
                    video_dir = os.path.join(self.embed_dir, video)
                    save_path = os.path.join(video_dir, "img%05d_%03d" % (fra, did))
                    np.save(save_path, f)

                    if not os.path.exists(os.path.join("XXX", video)):
                        os.mkdir(os.path.join("XXX", video))
                    video_dir = os.path.join("XXX", video)
                    save_path = os.path.join(video_dir, "img%05d_%03d" % (fra, did))
                    np.save(save_path, p_p)

                self.detect_imgs = []
                self.detect_infos = []

            if len(det) == 0:
                continue
            det[:, :4] = scale_coords(s[2:], det[:, :4], shape).round()
            
            det_idx = 0
            image_raw = copy.deepcopy(image_raw)

            if (video, frame) not in self.test_gt_bbox_dict:
                continue
            
            image_raw_ = copy.deepcopy(image_raw)
            gt_bbox_list = np.array(self.test_gt_bbox_dict[(video, frame)])

            for *xyxy, conf, cls in reversed(det):
                # p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                p_p = np.array([[(int(xyxy[0]) + int(xyxy[2]))/2.0, (int(xyxy[1]) + int(xyxy[3]))/2.0]])
                offset_dist = np.min(cdist(p_p, gt_bbox_list,metric='euclidean'))
                if offset_dist > self.gt_thre:
                    continue

                bbox = image_raw_[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2]),:]
                self.detect_imgs.append(self.transform(bbox))
                self.detect_infos.append([video, frame, det_idx, p_p])
                # cv2.rectangle(image_raw, p1, p2, colors(cls, True), thickness=self.lw, lineType=cv2.LINE_AA)
                det_idx += 1

            # detect_img = np.asarray(image_raw)
            # if not os.path.exists(os.path.join(self.visual_dir, video)):
            #     os.mkdir(os.path.join(self.visual_dir, video))
            # video_dir = os.path.join(self.visual_dir, video)
            # output_path = os.path.join(video_dir, "img%05d.jpg" % frame)
            # cv2.imwrite(output_path, detect_img)
        
        detect_imgs = torch.stack(self.detect_imgs, dim=0).cuda()
        feat = self.embedding(detect_imgs).cpu().numpy()
        for i in range(len(self.detect_infos)):
            v, fra, did, p_p = self.detect_infos[i]
            f = feat[i]
            if not os.path.exists(os.path.join(self.embed_dir, video)):
                os.mkdir(os.path.join(self.embed_dir, video))
            video_dir = os.path.join(self.embed_dir, video)
            save_path = os.path.join(video_dir, "img%05d_%03d" % (fra, did))
            np.save(save_path, f)

            if not os.path.exists(os.path.join("XXX", video)):
                os.mkdir(os.path.join("XXX", video))
            video_dir = os.path.join("XXX", video)
            save_path = os.path.join(video_dir, "img%05d_%03d" % (fra, did))
            np.save(save_path, p_p)

        self.detect_imgs = []
        self.detect_infos = []
            
    def forward(self, x, rs, v, f, m, augment=False, visualize=False):
        pred = self.detect(x, augment=augment, visualize=visualize)
        self.process_pred(pred, x.shape, rs, v, f, m)

def make_model(cfg):
    if cfg.MODEL.NAME == 'baseline':
        model = Baseline(cfg)
    
    return model