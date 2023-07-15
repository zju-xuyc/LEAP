import yaml
import argparse
import torch
import numpy as np
import random
import os
import datetime
import json
import cv2
from loguru import logger
from tools.data_prepare import get_label_details
from settings.settings import video_details
from settings import settings
from configs import cfg
from detector import Detector,Detector_origin
from inference import get_final_tuple_from_detector
import time
from evaluate import parse_intervals,evaluate_query_result,evaluated_object_recall

def getYaml(file_path):
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def clear_dir(cfgs):

    import shutil
    if not os.path.exists('./outputs/selected_frames'):
        os.mkdir('./outputs/selected_frames')
    else:
        shutil.rmtree('./outputs/selected_frames')
        os.mkdir('./outputs/selected_frames')

    if not os.path.exists('./outputs/selected_frames_origin'):
        os.mkdir('./outputs/selected_frames_origin')
    else:
        shutil.rmtree('./outputs/selected_frames_origin')
        os.mkdir('./outputs/selected_frames_origin')

    if not os.path.exists('./outputs/traj_compute_tmp'):
        os.mkdir('./outputs/traj_compute_tmp')
    else:
        shutil.rmtree('./outputs/traj_compute_tmp')
        os.mkdir('./outputs/traj_compute_tmp')

    if not os.path.exists('./outputs/reid/%s'%(cfgs["video_name"])):
        os.mkdir('./outputs/reid/%s'%(cfgs["video_name"]))
    else:
        shutil.rmtree('./outputs/reid/%s'%(cfgs["video_name"]))
        os.mkdir('./outputs/reid/%s'%(cfgs["video_name"]))

    if not os.path.exists('./outputs/traj_match'):
        os.mkdir('./outputs/traj_match')
    else:
        shutil.rmtree('./outputs/traj_match')
        os.mkdir('./outputs/traj_match')

def make_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/jackson.yaml')
    parser.add_argument('--weight', type=str, default='./weights/best.pt')
    parser.add_argument("-rw","--reid_weight",default=\
                            "XXX")

    parser.add_argument("--use_cluster",action="store_false") 
    parser.add_argument("--use_distill",action="store_false")  
    parser.add_argument("--use_filter",action="store_false")  
    parser.add_argument("--use_hand_label",action="store_false")
    parser.add_argument("--type",type=str,default="test")
    parser.add_argument("--adaptive_sample",action="store_false")
    parser.add_argument("--use_external_pattern",action="store_true")
    parser.add_argument("--cross_roads",action="store_true") 

    parser.add_argument("-vis","--visualize",action='store_false')  
    parser.add_argument("--active_log",action='store_true')
    parser.add_argument("-sr","--save_result",action="store_false",help="Save parsed result")
    parser.add_argument("-ld","--load",action='store_false',help="load file from dict")
    parser.add_argument("--use_label",action='store_true',help="Use preprocesed label instead of bytetracker")
    parser.add_argument("--use_mask",action='store_false',help="Use mask while parsing")
    parser.add_argument("--device",type=str,default='1')

    return parser

def run_session(cfgs,args,logger):

    dataset_object = get_label_details(cfgs,args,logger)
    label_parsed, tuple_dict, label_tuple_origin = dataset_object.get_label()

    logger.info("%s %s Parsed"%(cfgs["video_name"],args.type))
    video_details.gt_labels = label_parsed
    video_details.gt_tuple = tuple_dict
    video_details.gt_tuple_origin = label_tuple_origin
    
    if not args.load and cfgs["dataset_group"]=='blazeit':  # 所有有真实标签的都不需要打伪标签
        
        from initialize import initialize
        tracks, preprocess_tuple = initialize(cfgs,args,logger)
        
    else:
        try:
            if not args.use_external_pattern:
                tracks = np.load("./fixed_files/preprocessed/%s/"%(cfgs["video_name"])+cfgs["video_name"]+\
                    "_0_%d_tracks_clustered.npy"%(cfgs["start_frame"]),allow_pickle=True)
            else:
                tracks = np.load("./fixed_files/preprocessed/%s/"%(cfgs["video_name"])+cfgs["video_name"]+\
                    "_0_0_tracks_clustered.npy",allow_pickle=True)
                
        except Exception:
            logger.info("False Path:"+"./fixed_files/preprocessed/"+\
                cfgs["video_name"]+"_0_%d_tracks_clustered.npy"%(cfgs["start_frame"]))
            exit()

    # 明确目标检测器
    if args.use_distill:
        logger.info("Distilled")
        detector = Detector(args.weight,640,args.device,cfgs['iou_thresh'],cfgs['conf_thresh'])
    else:
        logger.info("Origin")
        detector = Detector_origin(cfg)
        detector.preprocess(cfg, [cfgs["h"],cfgs["w"]])
    
    if cfgs["dataset_group"] == "blazeit":
        video_path = os.path.join(settings.video_path,"standard_split",cfgs['full_name'],'concat',args.type,\
            '%s_%s.mp4'%(cfgs['video_name'],args.type))
    elif cfgs["dataset_group"] == "m30":
        video_path = os.path.join("XXX","%s.mp4"%(cfgs["video_name"]))
    

    if cfgs["dataset_group"]=="blazeit":
        args.reid_weight = os.path.join(args.reid_weight,cfgs['video_name'],"bagtricks_R50-ibn/model_best.pth")
    else:
        args.reid_weight = os.path.join(args.reid_weight,"jackson","bagtricks_R50-ibn/model_best.pth")

    frame_sampled, resolved_tuple = get_final_tuple_from_detector\
    (tracks,args.reid_weight,video_path,detector,cfgs,args,logger)
    
    logger.info("Sampled %d frames" % len(frame_sampled))
    logger.info("Filtered %d frames" % video_details.differencor)
    logger.info("Inference Time")
    logger.info(time.time()-video_details.start_time)
    logger.info("Detetor Time")
    logger.info(video_details.detector_time)
    logger.info("Differencor Time")
    logger.info(video_details.frame_differencer_time)
    logger.info("Reid Time")
    logger.info(video_details.reid_time)
    logger.info("Match Time")
    logger.info(video_details.match_time)
    logger.info("Decode Time")
    logger.info(video_details.decode_time)
    
    return_tuple = parse_intervals(resolved_tuple)

    if args.save_result:
        results = json.dumps(return_tuple)
        f_outputs = open('./outputs/parsed_results/%s_results.json'%(cfgs['video_name']),'w')
        f_outputs.write(results)
        f_outputs.close()
        
        f_outputs = open('./outputs/parsed_results/%s_results_dict.json'%(cfgs['video_name']),'w')
        results = json.dumps(video_details.match_dict)
        f_outputs.write(results)
        f_outputs.close()
        
    video_details.return_tuple = return_tuple

    evaluate_query_result(args,cfgs,label_tuple_origin)
    recall_by_sampled_frame = evaluated_object_recall(video_details.frame_sampled,tuple_dict)

    logger.info("Vehicle Recall: ")
    logger.info(recall_by_sampled_frame)
        
    
if __name__ =="__main__":

    parser = make_parser()
    args = parser.parse_args()
    cfgs = getYaml(args.config)
    logger.add('%s.log'%(datetime.datetime.now().strftime('%Y-%m-%d_%H')))
    set_seed(cfgs['seed'])
    clear_dir(cfgs)
    logger.info("Clean Folders")
    run_session(cfgs,args,logger)