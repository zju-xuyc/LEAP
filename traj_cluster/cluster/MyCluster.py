
from traj_cluster.cluster.distance import feature_cosine_distance
from traj_cluster.cluster.distance import feature_l2_distance

def MyCluster(length_list, feature_list, dis_threshold, clustering_mode):

    total_list = []
    for i in range(len(length_list)):
        total_list.append([length_list[i], i])
    total_list.sort(key = lambda x: x[0], reverse = True)
    
    cluster_id = 0
    cluster_label = [0] * len(feature_list)
    visit_flag = [0] * len(feature_list)
    for i in range(len(total_list)):
        if visit_flag[i] != 0:
            continue
        else:
            cluster_label[total_list[i][1]] = cluster_id
            cluster_id += 1
            candidate_list = [total_list[i][1]]
            while(len(candidate_list) != 0):
                current_id = candidate_list[0]
                candidate_list.remove(current_id)
                visit_flag[current_id] = 1
                for j in range(len(total_list)):
                    if visit_flag[j] != 0:
                        continue
                    else:
                        if "cosine" in clustering_mode:
                            tem_dis = feature_cosine_distance(feature_list[current_id], feature_list[j])
                        else:
                            tem_dis = feature_l2_distance(feature_list[current_id], feature_list[j])
                        if tem_dis < dis_threshold:
                            cluster_label[j] = cluster_id
                            candidate_list.append(j)
                            visit_flag[j] = 1
    return cluster_label