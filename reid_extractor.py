
from vehicle_reid.fastreid.config import get_cfg
import argparse
import numpy as np
import torch.nn.functional as F
import cv2
from vehicle_reid.predictor import FeatureExtractionDemo

def compute_cosine_similarity(vec1,vec2):
    cos_sim = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return cos_sim

def setup_cfg(args):
    
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    return cfg

class Parser(object):
    def __init__(self):

        self.FILE = "./reid_configs/VPDB/bagtricks_R50-ibn.yml"
        self.config_file = "./reid_configs/VPDB/bagtricks_R50-ibn.yml"
        self.parallel = False
        self.opts = []

    def change_yml_config(self,config_file):
        self.config_file = config_file


def get_parser():

    parser = Parser()    
    return parser

def postprocess(features):
    # Normalize feature to compute cosine distance
    features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features
        

class feature_extractor(object):
    
    def __init__(self, reid_weight,width,height):
        self.args = []
        self.extractor = []
        self.reid_weight = reid_weight
        self.width = width
        self.height = height
        
    def init_extractor(self,config_file="",):
        
        # args = get_parser().parse_args()
        args = get_parser()
        self.args = args
        if config_file != "":
            args.config_file = config_file
        cfg = setup_cfg(args)
        self.extractor = FeatureExtractionDemo(cfg, self.reid_weight, width=self.width, height=self.height,parallel=args.parallel)
        
    def inference_pic(self,img):
        score = self.extractor.run_on_images(img)
        # feat = postprocess(feat)
        return score

    def inference_feature(self,img):
        feature = self.extractor.run_on_image(img)
        # feat = postprocess(feat)
        return feature
    
if __name__ == "__main__":
    
    extractor = feature_extractor()
    extractor.init_extractor()
    pic1 = cv2.imread("./6_c1_82.jpg")
    pic2 = cv2.imread("./6_c1_127.jpg")
    vec1 = extractor.inference_pic(pic1)[0]
    vec2 = extractor.inference_pic(pic2)[0]
    print(compute_cosine_similarity(vec1,vec2))
        
             