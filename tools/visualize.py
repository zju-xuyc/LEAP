import cv2
import numpy as np
import os

def is_point_in_rectangle(point, rectangle):

    x, y = point
    left, top, right, bottom = rectangle
    return left <= x <= right and top <= y <= bottom

def allocate_tracks(tracks,stop_area):
    allocated_id = []
    for track in tracks:
        if len(track)<30 or len(track)>450:
            allocated_id.append([-1,-1])
            continue
        traj_pair=[]
        start_point = ((track[0][0]+track[0][2])/2,(track[0][1]+track[0][3])/2)
        end_point = ((track[-1][0]+track[-1][2])/2,(track[-1][1]+track[-1][3])/2)
        for i in range(len(stop_area)):
            flag = is_point_in_rectangle(start_point,stop_area[i])
            if flag:
                traj_pair.append(i)
                break
        for i in range(len(stop_area)):
            flag = is_point_in_rectangle(end_point,stop_area[i])
            if flag:
                traj_pair.append(i)
                break
        if len(traj_pair)<2:
            allocated_id.append([-1,-1])
        else:
            allocated_id.append(traj_pair)
    return allocated_id   

def draw_from_labels(video_name,video_labels,save_name,cfg,k=1000):
    tracks = [[] for i in range(2000)]
    for frame_id in range(len(video_labels)):
        detects = video_labels[frame_id]
        for detect in detects:
            tracks[detect[0]].append([detect[1],detect[2],detect[3],detect[4]])
    draw_trajectory(video_name,tracks[:k],save_name,cfg)
        

def draw_trajectory(video_name,tracks,save_name,cfg):

    if len(tracks)==2:
        tracks = tracks[0]
    colors = [(255,255,255),(0,0,255),(0,255,0),(255,0,0),(255,255,0),(0,100,255),(0,255,255),(100,100,255),(100,0,255),\
        (255,255,255),(0,0,255),(0,255,0),(255,0,0),(255,255,0),(0,100,255),(0,255,255),(100,100,255),(100,0,255)]
    video_obj = cv2.VideoCapture(video_name)
    video_img = video_obj.read()[1]
    count = -1
    tag=0
    stop_area = cfg["stop_area"]
    allocate_id = allocate_tracks(tracks,stop_area)
    allocate_color = {"[-1, -1]":(0,0,0)}
    for id in allocate_id:
        if str(id) not in allocate_color.keys():
            allocate_color[str(id)] = colors[tag]
            tag+=1
    for track in tracks:
        count+=1
        if len(track)<30 or len(track)>450:
            continue
        color = allocate_color[str(allocate_id[count])]

        for point in track:
            cv2.circle(video_img,(int(point[0]*0.5+point[2]*0.5),int(point[1]*0.5+point[3]*0.5)),2,color,-1)
            
    cv2.imwrite(save_name,video_img)

def draw_trajectory_from_track(tracks, cfg):

    tmp = cv2.imread(os.path.join("XXX",\
                                  cfg["video_name"])+".jpg")
    for track in tracks[0]:
        for point in track:
            cv2.circle(tmp,(int(point[0]*0.5+point[2]*0.5),int(point[1]*0.5+point[3]*0.5)),2,(255,255,255),-1)
    cv2.imwrite("/XXX/tmp_file/\
                 %s_tmp_tracks.jpg"%(cfg["video_name"]),tmp)
    
    draw_from_labels(video_path,label_parsed,'XXX/\
                     %s_traj_all.jpg'%(cfg["video_name"]),cfg,400)
    draw_from_labels(video_path,label_parsed,'XXX/\
                     %s_traj_intialize.jpg'%(cfg["video_name"]),cfg,80)
    
