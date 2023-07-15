from cProfile import label
import os
import cv2
import time
import torch
import numpy as np
import xml.dom.minidom
import matplotlib.pyplot as plt
import torchvision.transforms as T
from scipy.spatial.distance import cdist
from video_labeling.utils import non_max_suppression, scale_coords, calculate_p_r_f, letterbox_first, letterbox_second

def get_ignore_infos(video_ignore_path, image_shape):
    pads = np.zeros(image_shape)

    #打开xml文档
    dom = xml.dom.minidom.parse(video_ignore_path)

    #得到文档元素对象
    root = dom.documentElement
    bbs = [[float(bb.getAttribute("left")), float(bb.getAttribute("top")), float(bb.getAttribute("width")), float(bb.getAttribute("height"))] for bb in root.getElementsByTagName("ignored_region")[0].getElementsByTagName("box")]
    
    for bb in bbs:
        cv2.rectangle(pads, (int(bb[0]), int(bb[1])), (int(bb[0]+bb[2]), int(bb[1]+bb[3])), color=(255, 255, 255), thickness=-1)
    
    return pads

def get_label_infos(video_label_path, video_ignore_path, img_shape):
    gt_mask = get_ignore_infos(video_ignore_path, img_shape)
    targets, gt_info_dict = {}, {}
    with open(video_label_path, "r") as f:
        for info_line in f:
            info_line = info_line.rstrip("\n").split(" ")
            frame, x1, y1, width, height, vid = int(info_line[1]), float(info_line[4]), float(info_line[5]), float(info_line[6]), float(info_line[7]), int(info_line[8])

            if gt_mask[int(y1+height/2), int(x1+width/2)] == 255:
                continue

            if frame not in gt_info_dict:
                gt_info_dict[frame] = [[vid], 
                                       [[x1+width/2, y1+height/2]], 
                                       [[int(x1), int(y1), int(width), int(height)]]]
            else:
                gt_info_dict[frame][0].append(vid)
                gt_info_dict[frame][1].append([x1+width/2, y1+height/2])
                gt_info_dict[frame][2].append([int(x1), int(y1), int(width), int(height)])

            if vid not in targets:
                targets[vid] = [frame, frame]
            else:
                if frame < targets[vid][0]:
                    targets[vid][0] = frame
                if frame > targets[vid][1]:
                    targets[vid][1] = frame
    return targets, gt_info_dict

def get_area_label_infos(video_label_path, video_ignore_path, img_shape, areas=[[365, 0],[539, 737]]):
    gt_mask = get_ignore_infos(video_ignore_path, img_shape)
    targets, gt_info_dict = {}, {}
    with open(video_label_path, "r") as f:
        for info_line in f:
            info_line = info_line.rstrip("\n").split(" ")
            frame, x1, y1, width, height, vid = int(info_line[1]), float(info_line[4]), float(info_line[5]), float(info_line[6]), float(info_line[7]), int(info_line[8])

            if gt_mask[int(y1+height/2), int(x1+width/2)] == 255:
                continue
            
            if int(y1+height/2) < areas[0][0] or int(y1+height/2) > areas[1][0]:
                continue

            if int(x1+width/2) < areas[0][1] or int(x1+width/2) > areas[1][1]:
                continue

            if frame not in gt_info_dict:
                gt_info_dict[frame] = [[vid], [[x1+width/2, y1+height/2 - 365]], [[int(x1), int(y1), int(width), int(height)]]]
            else:
                gt_info_dict[frame][0].append(vid)
                gt_info_dict[frame][1].append([x1+width/2, y1+height/2 - 365])
                gt_info_dict[frame][2].append([int(x1), int(y1), int(width), int(height)])

            if vid not in targets:
                targets[vid] = [frame, frame]
            else:
                if frame < targets[vid][0]:
                    targets[vid][0] = frame
                if frame > targets[vid][1]:
                    targets[vid][1] = frame
    return targets, gt_info_dict

def detect_process(img_raw, stride, pt, cfg, model, frame, 
                   new_unpad=None, shape=None, top=None, bottom=None, left=None, right=None):
    time_start = time.time()
    if frame == 0:
        img, new_unpad, shape, top, bottom, left, right = letterbox_first(img_raw, cfg.INPUT.SIZE, stride=stride, auto=pt)
        img = (torch.FloatTensor(np.ascontiguousarray(img.transpose((2, 0, 1))[::-1]))/255.0).cuda().unsqueeze(0)
        pred = model.detect(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, cfg.MODEL.DETECT_CONF_THRE, cfg.MODEL.DETECT_IOU_THRE, cfg.MODEL.DETECT_CLASSES, cfg.MODEL.DETECT_AGNOSTIC_NMS, max_det=cfg.MODEL.DETECT_MAX_DETECT)
        det = pred[0]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_raw.shape).round()
        det_idx = 0
        detect_imgs, detect_infos = [], []
        for *xyxy, conf, cls in reversed(det):
            p_p = np.array([[(int(xyxy[0]) + int(xyxy[2]))/2.0, (int(xyxy[1]) + int(xyxy[3]))/2.0]])
            bbox = img_raw[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2]),:]
            detect_imgs.append(bbox)
            detect_infos.append([det_idx, p_p, xyxy])
            det_idx += 1
        
        time_end = time.time()
        return detect_imgs, detect_infos, time_end - time_start, new_unpad, shape, top, bottom, left, right
    else:
        img = letterbox_second(img_raw, new_unpad, shape, top, bottom, left, right)
        img = (torch.FloatTensor(np.ascontiguousarray(img.transpose((2, 0, 1))[::-1]))/255.0).cuda().unsqueeze(0)
        pred = model.detect(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, cfg.MODEL.DETECT_CONF_THRE, cfg.MODEL.DETECT_IOU_THRE, cfg.MODEL.DETECT_CLASSES, cfg.MODEL.DETECT_AGNOSTIC_NMS, max_det=cfg.MODEL.DETECT_MAX_DETECT)
        det = pred[0]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_raw.shape).round()
        det_idx = 0
        detect_imgs, detect_infos = [], []
        for *xyxy, conf, cls in reversed(det):
            p_p = np.array([[(int(xyxy[0]) + int(xyxy[2]))/2.0, (int(xyxy[1]) + int(xyxy[3]))/2.0]])
            bbox = img_raw[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2]),:]
            detect_imgs.append(bbox)
            detect_infos.append([det_idx, p_p, xyxy])
            det_idx += 1
        
        time_end = time.time()
        return detect_imgs, detect_infos, time_end - time_start

def embedding_process(detect_imgs, transforms, model):
    with torch.no_grad():
        time_start = time.time()
        detect_imgs = torch.stack([transforms(T.ToPILImage()(img).convert('RGB')) for img in detect_imgs], dim=0).cuda()
        feats = model.embedding(detect_imgs).cpu().numpy()
        time_end = time.time()
        return feats, time_end - time_start

def save_process_result(vid, gt_info_dict, frames, vehicle, save_dir, img_raw_dir, mask):
    vid_path = os.path.join(save_dir, str(vid))
    if not os.path.exists(vid_path):
        os.mkdir(vid_path)
    
    gt_images_path = os.path.join(vid_path, "gt")
    if not os.path.exists(gt_images_path):
        os.mkdir(gt_images_path)

    for frame in frames:
        img_path = os.path.join(img_raw_dir, "img%05d.jpg" % frame)
        if vid not in gt_info_dict[frame][0]:
            continue
        bbox = gt_info_dict[frame][2][gt_info_dict[frame][0].index(vid)]
        img_raw = cv2.imread(img_path)
        # img_raw[mask == 0, :] = 0
        # crop = img_raw[bbox[1]:(bbox[1]+bbox[3]), bbox[0]:(bbox[0]+bbox[2]), :]
        # print(img_raw.shape, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 2)
        cv2.rectangle(img_raw, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 2)
        cv2.imwrite(gt_images_path + "/img%05d_%05d.jpg" %(frame, vid), img_raw)

    pr_images_path = os.path.join(vid_path, "pr")
    if not os.path.exists(pr_images_path):
        os.mkdir(pr_images_path)
    
    for frame, xyxy in zip(vehicle.frame_list, vehicle.bboxs):
        img_path = os.path.join(img_raw_dir, "img%05d.jpg" % frame)
        img_raw = cv2.imread(img_path)
        # img_raw[mask == 0, :] = 0
        # crop = img_raw[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2]),:]
        cv2.rectangle(img_raw, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 0, 255), 2)
        cv2.imwrite(pr_images_path + "/img%05d_%05d.jpg" %(frame, vid), img_raw)

def test_result(final_vehicle_list, gt_info_dict, targets, video_name, sample_rate, image_raw_dir, mask):
    save_sample_dir = os.path.join("runs", sample_rate)
    if not os.path.exists(save_sample_dir):
        os.mkdir(save_sample_dir)

    save_vid_dir = os.path.join(save_sample_dir, video_name)
    if not os.path.exists(save_vid_dir):
        os.mkdir(save_vid_dir)

    singal_ps, singal_rs, singal_fs = [], [], []

    hit_vehicle, hit_vehicle_result = [], {}
    for vehicle in final_vehicle_list:   
        pending_vids = []
        for frame, p_bbox in zip(vehicle.frame_list, vehicle.bbox_list):
            if frame in gt_info_dict:
                gt_position = gt_info_dict[frame][1]
                gt_position = np.array(gt_position)
                vids = gt_info_dict[frame][0]
            else:
                for frame in reversed(vehicle.frame_list[:-1]):
                    if frame in gt_info_dict:
                        gt_position = gt_info_dict[frame][1]
                        gt_position = np.array(gt_position)
                        vids = gt_info_dict[frame][0]
                        break

            offset_dist = np.argmin(cdist(p_bbox, gt_position, metric='euclidean')[0])
            vid = vids[offset_dist]
            if cdist(p_bbox, gt_position, metric='euclidean')[0][offset_dist] > 20:
                continue
            pending_vids.append(vid)

        if len(pending_vids) == 0:
            continue
        elif len(pending_vids) == 1:
            vid = pending_vids[0]
        else:
            vid = max(pending_vids, key=pending_vids.count)

        target = targets[int(vid)]
        frame_bound = vehicle.frame_bound

        # save_process_result(vid, gt_info_dict, list(range(target[0], target[-1] + 1, 25)), vehicle, save_vid_dir, image_raw_dir, mask)

        p, r, f = calculate_p_r_f(frame_bound, target)

        # singal_ps.append(p)
        # singal_rs.append(r)
        # singal_fs.append(f)

        if vid not in hit_vehicle:
            hit_vehicle.append(vid)
            hit_vehicle_result[vid] = [[p], [r], [f], [vehicle]]
        else:
            hit_vehicle_result[vid][0].append(p)
            hit_vehicle_result[vid][1].append(r)
            hit_vehicle_result[vid][2].append(f)
            hit_vehicle_result[vid][3].append(vehicle)

    for vid in hit_vehicle_result:
        rid = np.argmax(hit_vehicle_result[vid][2])
        singal_ps.append(hit_vehicle_result[vid][0][rid])
        singal_rs.append(hit_vehicle_result[vid][1][rid])
        singal_fs.append(hit_vehicle_result[vid][2][rid])
        
        target = targets[vid]

        vid_path = os.path.join(save_vid_dir, str(vid))
        if not os.path.exists(vid_path):
            os.mkdir(vid_path)
        result_path = open(os.path.join(vid_path, "result.txt"), "w")
        result_path.write("Target:[%d, %d]  Predict:[%d, %d]\n" % (target[0], target[1], hit_vehicle_result[vid][3][rid].frame_bound[0], hit_vehicle_result[vid][3][rid].frame_bound[1]))
        result_path.write("Precision: %.4f, Recall: %.4f, F1-Score: %.4f" %(hit_vehicle_result[vid][0][rid], 
                                                                            hit_vehicle_result[vid][1][rid], 
                                                                            hit_vehicle_result[vid][2][rid]))

        save_process_result(vid, gt_info_dict, 
                            list(range(target[0], target[-1] + 1, 25)),
                            hit_vehicle_result[vid][3][rid], save_vid_dir, image_raw_dir, mask)

    return singal_ps, singal_rs, singal_fs, len(hit_vehicle)

def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.t())
    area2 = box_area(boxes2.t())

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = (rb - lt).clamp(min=0).prod(2)  # [N,M]
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def test_result_track(final_vehicle_list, gt_info_dict, targets):
    singal_ps, singal_rs, singal_fs = [], [], []

    hit_vehicle, hit_vehicle_result = [], {}
    for vehicle in final_vehicle_list:   
        track_data = final_vehicle_list[vehicle][1]
        pending_vids = []
        for x1, y1, x2, y2, frame in track_data:
            if frame in gt_info_dict:
                vids = gt_info_dict[frame][0]
                gt_bboxs = torch.FloatTensor([[x, y, x+w, y+h] for (x, y, w, h) in gt_info_dict[frame][2]]).cuda()
            # else:
            #     continue
            vbbox = torch.FloatTensor([[x1, y1, x2, y2]]).cuda()

            iou = int(torch.argmax(box_iou(vbbox, gt_bboxs)[0]))
            vid = vids[iou]
            pending_vids.append(vid)

        if len(pending_vids) == 0:
            continue
        elif len(pending_vids) == 1:
            vid = pending_vids[0]
        else:
            vid = max(pending_vids, key=pending_vids.count)

        target = targets[int(vid)]
        frame_bound = final_vehicle_list[vehicle][0]

        p, r, f = calculate_p_r_f(frame_bound, target)

        if vid not in hit_vehicle:
            hit_vehicle.append(vid)
            hit_vehicle_result[vid] = [[p], [r], [f], [vehicle]]
        else:
            hit_vehicle_result[vid][0].append(p)
            hit_vehicle_result[vid][1].append(r)
            hit_vehicle_result[vid][2].append(f)
            hit_vehicle_result[vid][3].append(vehicle)
    
    for vid in hit_vehicle_result:
        rid = np.argmax(hit_vehicle_result[vid][2])
        singal_ps.append(hit_vehicle_result[vid][0][rid])
        singal_rs.append(hit_vehicle_result[vid][1][rid])
        singal_fs.append(hit_vehicle_result[vid][2][rid])

    return singal_ps, singal_rs, singal_fs, len(hit_vehicle)

def test_singal_result(track_data, gt_info_dict):
    pending_vids = []
    for x1, y1, x2, y2, frame in track_data:
        if frame in gt_info_dict:
            vids = gt_info_dict[frame][0]
            gt_bboxs = torch.FloatTensor([[x, y, x+w, y+h] for (x, y, w, h) in gt_info_dict[frame][2]]).cuda()
        # else:
        #     continue
        vbbox = torch.FloatTensor([[x1, y1, x2, y2]]).cuda()

        iou = int(torch.argmax(box_iou(vbbox, gt_bboxs)[0]))
        vid = vids[iou]
        pending_vids.append(vid)

    if len(pending_vids) == 0:
        vid = -1
    elif len(pending_vids) == 1:
        vid = pending_vids[0]
    else:
        vid = max(pending_vids, key=pending_vids.count)
    
    return vid

def analysis_time(gt_info_dict, targets):
    print("Analysis Time !")
    save_dir = "XXX"
    for vid in targets:
        frame_bound = targets[vid]
        vxs, vys, vs, fs = [], [], [], []
        begin_frame, begin_position = frame_bound[0], gt_info_dict[frame_bound[0]][1][gt_info_dict[frame_bound[0]][0].index(vid)]
        for frame in range(frame_bound[0] + 1, frame_bound[1] + 1):
            if frame in gt_info_dict:
                vids = gt_info_dict[frame][0]
                cxys = gt_info_dict[frame][1]
                idx = vids.index(vid)
                cxy = cxys[idx]
                vx = abs(cxy[0] - begin_position[0]) / (frame - begin_frame)
                vy = abs(cxy[1] - begin_position[1]) / (frame - begin_frame)
                v = (vx**2 + vy**2)**0.5
                vxs.append(vx)
                vys.append(vy)
                vs.append(v)
                begin_frame = frame
                begin_position = cxy
                fs.append(frame)

        if len(fs) > 6:
            tts = fs[1:len(fs):(len(fs) - 1)//5]
            tvs = vs[1:len(fs):(len(fs) - 1)//5]

        plt.plot(list(range(len(vxs)))[10:], vxs[10:], label="vxs")
        plt.plot(list(range(len(vys)))[10:], vys[10:], label="vys")
        plt.plot(list(range(len(vs)))[10:], vs[10:], label="vs")
        plt.legend()
        savefig_path = os.path.join(save_dir, "%d.jpg" % vid)
        plt.savefig(savefig_path)
        plt.cla()
    print("Analysis Time Done !")