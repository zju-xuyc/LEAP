import os
import cv2
import random
import argparse
import numpy as np

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

def track_remap(before_gts):
    after_gts = {}
    track_id_map = {}
    for track_id in before_gts:
        if track_id not in track_id_map:
            track_id_map[track_id] = len(track_id_map) + 1
        after_gts[track_id_map[track_id]] = before_gts[track_id]
    del before_gts
    return after_gts

def post_process_deletete_by_noise(before_gts, frame_gap=60):
    track_ids = list(before_gts.keys())
    for track_id in track_ids:
        if before_gts[track_id]['frame_gap'] < frame_gap:
            del before_gts[track_id]
    after_gts = track_remap(before_gts)
    return after_gts

def post_process_deletete_by_position(before_gts, min_size=20):
    for track_id in before_gts:
        if before_gts[track_id]['size'] < min_size:
            del before_gts[track_id]
    after_gts = track_remap(before_gts)
    return after_gts

def load2gts(read_path):
    gts = {}
    with open(read_path, "r") as reader:
        for line in reader:
            frame_id,tid,x,y,w,h,score,c_id,_,_,_ = line.rstrip("/n").split(",")
            frame_id = int(frame_id)
            tid = int(tid)
            x = float(x)
            y = float(y)
            w = float(w)
            h = float(h)
            score = float(score)
            c_id = float(c_id)
            if tid not in gts:
                gts[tid] = {}
                gts[tid]['frame_bound'] = [99999, -99999]
                gts[tid]['frame_gap'] = 0
                gts[tid]['points'] = [[frame_id,tid,x,y,w,h,score,c_id]]
            else:
                gts[tid]['points'].append([frame_id,tid,x,y,w,h,score,c_id])
                if frame_id < gts[tid]['frame_bound'][0]:
                    gts[tid]['frame_bound'][0] = frame_id
                if frame_id > gts[tid]['frame_bound'][1]:
                    gts[tid]['frame_bound'][1] = frame_id
                gts[tid]['frame_gap'] = gts[tid]['frame_bound'][1] - gts[tid]['frame_bound'][0]
    return gts

def gts2fts(gts):
    fts = {}
    for trackid in gts:
        points = gts[trackid]['points']
        for point in points:
            frame_id,tid,x,y,w,h,score,c_id = point
            if frame_id not in fts:
                fts[frame_id] = [[[x,y,w,h]], 
                                 [tid]]
            else:
                fts[frame_id][0].append([x,y,w,h])
                fts[frame_id][1].append(tid)
    return fts
    
def write_post_result(gts, write_path):
    with open(write_path, "w") as writer:
        for tid in gts:
            for point in gts[tid]['points']:
                writer.write(",".join([str(i) for i in point]) + ",-1,-1,-1\n")
                
def write_post_video_result(gts, input_video_path, write_video_path):
    fts = gts2fts(gts)
    
    videoCapture = cv2.VideoCapture(input_video_path)
    frame_count = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    vid_writer = cv2.VideoWriter(write_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    
    frame_id = 0
    success, image0 = videoCapture.read()
    
    while success:
        if frame_id not in fts:
            success, image0 = videoCapture.read()
            online_im = image0
        else:
            online_tlwhs, online_ids = fts[frame_id]
            online_im = plot_tracking(
                image0, online_tlwhs, online_ids,
                frame_id + 1, fps=56 + random.random() * 2.5
                )
        if frame_id % 1000 == 0:
            print("frame_id: ", frame_id)
        vid_writer.write(online_im)
        success, image0 = videoCapture.read()
        frame_id += 1
    videoCapture.release()
    vid_writer.release()

if __name__ == "__main__":    
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("--read_path", type=str, default="./videos_result/new_tmp/XXX.txt", help="video path")
    parser.add_argument("--write_path", type=str, default="./videos_result/new_tmp/XXX.txt", help="video path")
    parser.add_argument("--video_path", type=str, default="./videos/XXX.mp4", help="save video path")
    parser.add_argument("--write_video_path", type=str, default="./videos_result/new_tmp/XXX.mp4", help="save video path")
    args = parser.parse_args()
    
    read_gts = load2gts(args.read_path)
    processbygaps_gts = post_process_deletete_by_noise(read_gts, frame_gap=60)
    
    write_post_result(processbygaps_gts, args.write_path)
    write_post_video_result(processbygaps_gts, args.video_path, args.write_video_path)