import os
import cv2
import time
import torch
import pickle
import random
import argparse
import numpy as np
from video_labeling.config import cfg
from video_labeling.timer import Timer
from loguru import logger
import matplotlib.pyplot as plt
from video_labeling.models import load_yolo_detection
from video_labeling.tracker.byte_tracker import BYTETracker, BaseTrack
from video_labeling.video_process import get_label_infos, test_result_track, test_singal_result, analysis_time
from video_labeling.utils import letterbox_first, letterbox_second, non_max_suppression, scale_coords

def plot_tracking(image, tlwhs, obj_ids, scores=None, frame_id=0, fps=0., ids2=None):
   def get_color(idx):
      idx = idx * 3
      color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

      return color

   im = np.ascontiguousarray(np.copy(image))
   im_h, im_w = im.shape[:2]

   top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

   text_scale = 2
   text_thickness = 2
   line_thickness = 3

   radius = max(5, int(im_w/140.))
   cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
               (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

   for i, tlwh in enumerate(tlwhs):
      x1, y1, w, h = tlwh
      intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
      obj_id = int(obj_ids[i])
      id_text = '{}'.format(int(obj_id))
      if ids2 is not None:
         id_text = id_text + ', {}'.format(int(ids2[i]))
      color = get_color(abs(obj_id))
      cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
      cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                  thickness=text_thickness)

   return im

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("-demo", default="video", help="demo type, eg. image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument(
        "--path", default="./videos", help="path to images or video"
    )

    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_false",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--sampling_rate", default=1, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=5, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument('--detect_conf_thre', type=float, default=0.1, help='detect confidence threshold')
    parser.add_argument("--video_path", type=str, default="XXX.mp4", help="video path")
    parser.add_argument("--mask_path", type=str, default="./masks/XXX.jpg", help="video path")
    parser.add_argument("--save_path", type=str, default="./videos_result/new_tmp", help="save video path")
    parser.add_argument("--video_name", type=str, default="square_val", help="video name")
    return parser

def set_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True

class Detector():
   def __init__(self, cfg):

        self.model, self.stride, self.pt = load_yolo_detection(cfg)
        self.model.eval()
        self.detactor_time = []

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
            pred = self.model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, cfg.MODEL.DETECT_CONF_THRE, cfg.MODEL.DETECT_IOU_THRE, cfg.MODEL.DETECT_CLASSES, cfg.MODEL.DETECT_AGNOSTIC_NMS, max_det=cfg.MODEL.DETECT_MAX_DETECT)
        det = pred[0]
        # [x1,y1,x2,y2,conf,class]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], self.img0.shape).round()
        detect_infos = [None for _ in range(len(det))]
        det_idx = 0
        for *xyxy, conf, class_label in reversed(det):
            detect_infos[det_idx] = torch.Tensor([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), conf, int(class_label)])
            det_idx += 1
        if len(detect_infos) == 0:
            return [], False
        detect_infos = torch.stack(detect_infos, dim=0)
        t2 = time.time()
        self.detactor_time.append(t2 - t1)
        return detect_infos, True
    
def labeling_video(args):
    """
    Video Pseudo Labeling
    video_path: input video path [str]
    save_path: pseudo label save path
    video_name: name to save the input video
    """
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)
    # os.environ["CUDA_VISIBLE_DEVICES"] = cfg.MODEL.DEVICE_ID
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")
    
    time_all = []
    
    detector = Detector(cfg) # 载入目标检测器
    timer = Timer()
    
    videoCapture = cv2.VideoCapture(args.video_path)
    success, image0 = videoCapture.read()
    
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    labeled_video_save_path = os.path.join(args.save_path, args.video_name+"_annotated.mp4")
    vid_writer = cv2.VideoWriter(labeled_video_save_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
    
    detector.preprocess(cfg, [size[1], size[0]])
    results, frame_id, tracks_result = [], 0, {}
    mask_read = cv2.imread(args.mask_path, 0)
    
    while success:
        image0 = cv2.add(image0, np.zeros(np.shape(image0), dtype=np.uint8), mask=mask_read)

        dets, flag = detector(image0, timer)

        if frame_id == 0:
            tracker = BYTETracker(args)
            
        if flag:
            if frame_id % 30 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
            
            t1 = time.time()
            online_targets = tracker.update(dets, [1, 1], [1, 1])
            t2 = time.time()
            # 记录跟踪时间
            time_all.append([detector.detactor_time[-1], t2 - t1, frame_id])

            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_velocity = []
            online_c_ids = []
                
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                xv, yv = t.mean[4], t.mean[5]
                if tlwh[2] * tlwh[3] > args.min_box_area: # and not vertical:
                    online_tlwhs.append(tlwh) 
                    online_ids.append(tid)    
                    online_scores.append(t.score) 
                    online_velocity.append([xv, yv]) 
                    online_c_ids.append(t.c_id_list[-1]) 
                    results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},{t.c_id_list[-1]:.2f},-1,-1,-1\n"
                    )
            timer.toc()
            for oid, otlwh, ovxvy, ocid in zip(online_ids, online_tlwhs, online_velocity, online_c_ids):
                x1, y1, w, h = otlwh
                if oid not in tracks_result:
                    tracks_result[oid] = [[frame_id, frame_id], [[x1, y1, x1+w, y1+h, frame_id,ocid]], [ovxvy]]
                else:
                    if frame_id < tracks_result[oid][0][0]:
                        tracks_result[oid][0][0] = frame_id
                    if frame_id > tracks_result[oid][0][1]:
                        tracks_result[oid][0][1] = frame_id
                    tracks_result[oid][1].append([x1, y1, x1+w, y1+h, frame_id,ocid])
                    tracks_result[oid][2].append(ovxvy)

            online_im = plot_tracking(
            detector.img0, online_tlwhs, online_ids,
            frame_id = frame_id + 1, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = detector.img0
        if args.save_result:
            vid_writer.write(online_im)
        
        success, image0 = videoCapture.read()
        frame_id += 1
            
    videoCapture.release()
    pickle.dump(tracks_result, open(args.save_path + "/label_%s.pkl" % (args.video_name),"wb"))
    
    with open(os.path.join(args.save_path, "label_%s.txt" % (args.video_name)),"w") as result_f:
        for result in results:
            result_f.write(result)
            
    result_f.close()
        
    del tracker, detector
    BaseTrack._count = 0

    return tracks_result
    
def get_k_frame_label(video_path,mask_image,k=1000):

    args = make_parser().parse_args()
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)
    args.device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    
    time_all = []
    
    detector = Detector(cfg)
    timer = Timer()

    videoCapture = cv2.VideoCapture(video_path)
    success, image0 = videoCapture.read()
    
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    detector.preprocess(cfg, [size[1], size[0]])
    results, frame_id, tracks_result = [], 0, {}
    mask_read = mask_image

    while success:
        image0 = cv2.add(image0, np.zeros(np.shape(image0), dtype=np.uint8), mask=mask_read)
        background_img = image0
        dets, flag = detector(image0, timer)
        if frame_id == 0:
            tracker = BYTETracker(args)
            
        if frame_id<k:

            if frame_id % 30 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
            if flag:
                t1 = time.time()
                online_targets = tracker.update(dets, [1, 1], [1, 1])
                t2 = time.time()
                time_all.append([detector.detactor_time[-1], t2 - t1, frame_id])

                online_tlwhs = []
                online_ids = []
                online_scores = []
                online_velocity = []
                online_c_ids = []
                    
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    xv, yv = t.mean[4], t.mean[5]
                    if tlwh[2] * tlwh[3] > args.min_box_area: # and not vertical:
                        online_tlwhs.append(tlwh) 
                        online_ids.append(tid)    
                        online_scores.append(t.score) 
                        online_velocity.append([xv, yv]) 
                        online_c_ids.append(t.c_id_list[-1]) 
                        results.append(
                                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},{t.c_id_list[-1]:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                for oid, otlwh, ovxvy, ocid in zip(online_ids, online_tlwhs, online_velocity, online_c_ids):
                    x1, y1, w, h = otlwh
                    if oid not in tracks_result:
                        tracks_result[oid] = [[frame_id, frame_id], [[x1, y1, x1+w, y1+h, frame_id,ocid]], [ovxvy]]
                    else:
                        if frame_id < tracks_result[oid][0][0]:
                            tracks_result[oid][0][0] = frame_id
                        if frame_id > tracks_result[oid][0][1]:
                            tracks_result[oid][0][1] = frame_id
                        tracks_result[oid][1].append([x1, y1, x1+w, y1+h, frame_id,ocid])
                        tracks_result[oid][2].append(ovxvy)
            else:
                timer.toc()
        else:
            timer.toc()
            break
        
        success, image0 = videoCapture.read()
        if not success:
            print("video capture failed")
            print(frame_id)
        frame_id += 1
            
    videoCapture.release()
        
    del tracker, detector
    BaseTrack._count = 0
    return tracks_result,background_img

if __name__ == "__main__":
    args = make_parser().parse_args()
    labeling_video(args)