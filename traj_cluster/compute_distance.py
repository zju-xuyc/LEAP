import os
import time
import pickle
import numpy as np
import multiprocessing
from distance import distance

def trajectory_all_distance(tr_id, trs_compare_dict):
    print("Begin compute")

    if not os.path.exists("./distance_compution/frechet_distance"):
        os.makedirs("./distance_compution/frechet_distance")
    trs_matrix = distance.cdist(trs_compare_dict, {tr_id:trs_compare_dict[tr_id]},type="frechet")
    pickle.dump(trs_matrix, open('./distance_compution/frechet_distance/frechet_distance_'+str(tr_id), 'wb'))
    
    print("Complete: "+str(tr_id))

def read_points(path):
    traj_point_list = []
    begin_second = int(time.mktime(time.strptime("2020-05-31 23:59:00", "%Y-%m-%d %H:%M:%S")))
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            line_list = line.split('\t')
            current_time = line_list[1] + " " + line_list[2]
            current_time_stamp = int(time.mktime(time.strptime(current_time, "%Y-%m-%d %H:%M:%S"))) - begin_second
            lon = float(line_list[3])
            lat = float(line_list[4])
            traj_point_list.append([current_time_stamp, lon, lat])
    return traj_point_list

def get_total_traj_dict(root_read_path):
    traj_dict = {}
    files = os.listdir(root_read_path)
    for file in files:
        if "txt" not in file:
            continue
        id = int(file.split('.')[0].split("-")[1])
        read_path = root_read_path + file
        points = read_points(read_path)
        traj_dict[id] = points
    return traj_dict


def compute_distance():
    trjs = get_total_traj_dict("XXX")
    trs_compare_dict = {}
    for tr_id in trjs:
        tr = trjs[tr_id]
        trarray = []
        for record in tr:
            trarray.append([record[1],record[2]])
        trs_compare_dict[tr_id] = trarray
    pool = multiprocessing.Pool(processes=80)
    for tr_id in trs_compare_dict:
        pool.apply_async(trajectory_all_distance, (tr_id, trs_compare_dict))
    pool.close()
    pool.join()


def combainDistances(inputPath = './distance_compution/DTW_distance/'):
    files = os.listdir(inputPath)
    files_index = []
    for fn in files:
        i = int(fn.split('_')[2])
        files_index.append((fn,i))
    files_index.sort(key=lambda x:x[1])
    distances = {}
    for fn in files_index:
        dis = pickle.load(open(inputPath+fn[0], "rb"))
        for id in dis:
            distances[id] = dis[id]
    pickle.dump(distances,open('./distance_compution/'+inputPath.split('/')[2]+'_dict','wb'))


if __name__ == "__main__":
    compute_distance()
    combainDistances(inputPath='./distance_compution/frechet_distance/')
