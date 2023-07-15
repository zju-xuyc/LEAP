import numpy as np
from random import randint

from traj_cluster.cluster.distance import feature_cosine_distance

def update_center(assignments, centers):
    new_centers = []
    for center in assignments:
        slaves = assignments[center]
        if len(slaves) == 0:
            new_centers.append(centers[center])
        else:
            new_centers.append(np.mean(slaves, axis = 0).tolist())
    return new_centers

def assign_cluster(centers, feature_list, clustering_mode):

    assert len(centers) > 0, "Cluster Num > 0"

    assignments = {}
    for i, center in enumerate(centers):
        assignments[i] = []

    for seq_id in range(len(feature_list)):
        minimum_dis = 1000000
        minimum_index = 0
        for i in range(len(centers)):
            feature_0 = feature_list[seq_id]
            feature_1 = centers[i]
            if "cosine" in clustering_mode:
                distance_value = feature_cosine_distance(feature_0, feature_1)
            else:
                distance_value = 0
            if distance_value < minimum_dis:
                minimum_dis = distance_value
                minimum_index = i
        assignments[minimum_index].append(feature_list[seq_id])
    return assignments


def random_generate_k_centers(k, feature_list):
    centers = []
    for i in range(k):
        centers.append(feature_list[randint(0, len(feature_list)-1)])
    return centers

def Kmeans(n_clusters, feature_list, clustering_mode):
    
    assert n_clusters > 0, "k>0"

    feature_list = feature_list.tolist()

    centers = random_generate_k_centers(n_clusters, feature_list)
    centers.sort()

    assignments = assign_cluster(centers, feature_list, clustering_mode)
    iteration_num = 0
    while True:
        new_centers = update_center(assignments, centers)
        new_centers.sort()
        if (new_centers == centers) or (iteration_num == 20):
            break
        centers = new_centers
        assignments = assign_cluster(centers, feature_list, clustering_mode)
        iteration_num += 1
    labels = [0] * len(feature_list)
    cluster_id = 0
    for center in assignments:
        for slave in assignments[center]:
            labels[feature_list.index(slave)] = cluster_id
        cluster_id += 1
    return labels


if __name__ == "__main__":
    pass
