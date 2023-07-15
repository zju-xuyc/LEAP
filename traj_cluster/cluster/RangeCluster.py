
from traj_cluster.cluster.distance import feature_cosine_distance
from traj_cluster.cluster.distance import feature_l2_distance

def RangeCluster(length_list, feature_list, dis_threshold, clustering_mode):
    total_list = []
    for i in range(len(length_list)):
        total_list.append([length_list[i], i])
    total_list.sort(key = lambda x: x[0], reverse = True)

    cluster_id = 0
    cluster_center = []
    cluster_dict = {}
    cluster_label = [0] * len(feature_list)
    for i in range(len(total_list)):
        if i == 0:
            cluster_center.append(total_list[i][1])
            cluster_dict[total_list[i][1]]  = cluster_id
            cluster_label[total_list[i][1]] = cluster_id
            cluster_id += 1
        else:
            flag = 0
            for center in cluster_center:
                if "cosine" in clustering_mode:
                    tem_dis = feature_cosine_distance(feature_list[total_list[i][1]], feature_list[center])
                else:
                    tem_dis = feature_l2_distance(feature_list[total_list[i][1]], feature_list[center])
                if tem_dis < dis_threshold:
                    cluster_label[total_list[i][1]] = cluster_dict[center]
                    flag = 1
                    break
            if flag == 0:
                cluster_center.append(total_list[i][1])
                cluster_dict[total_list[i][1]] = cluster_id
                cluster_label[total_list[i][1]] = cluster_id
                cluster_id += 1
    return cluster_label