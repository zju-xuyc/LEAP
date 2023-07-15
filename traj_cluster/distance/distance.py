import numpy as np
from traj_cluster.distance.dtw import dtw
from traj_cluster.distance.frechet import frechet
from traj_cluster.distance.edr import edr
from traj_cluster.distance.hausdorff import hausdorff

def pdist(seq1, seq2, type, eps = 200):
    if type == "hausdorff":
        seq1 = np.array(seq1)
        seq2 = np.array(seq2)
        return hausdorff(seq1, seq2)
    elif type == "lcss":
        # return lcss(seq1, seq2, eps)
        return -1
    elif type == "edr":
        return edr(seq1, seq2, eps)
    elif type == "dtw":
        seq1 = np.array(seq1)
        seq2 = np.array(seq2)
        return dtw(seq1, seq2)
    elif type == "frechet":
        return frechet(seq1, seq2)
    else:
        raise ValueError("type error")

def cdist(traj_dict1, traj_dict2, type, eps = 200):
    dis_dict = {}
    for traj_id1 in traj_dict1:
        for traj_id2 in traj_dict2:
            if traj_id1>=traj_id2:
            # if traj_id1>=0:
                seq1 = traj_dict1[traj_id1]
                seq2 = traj_dict2[traj_id2]
                tem_dis = pdist(seq1, seq2, type, eps)
                dis_dict[(traj_id1, traj_id2)] = tem_dis
    return dis_dict
