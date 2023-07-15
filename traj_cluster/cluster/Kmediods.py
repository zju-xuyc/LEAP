from random import randint

def update_center(assignments, centers, distance_matrix, seq_id_list):
    new_centers = []
    for center in centers:
        slaves = assignments[center]
        minimum_dis = 1000000
        minimum_index = center
        for i in range(len(slaves)):
            slave = slaves[i]
            tem_sum = 0
            for j in range(len(slaves)):
                slave_ = slaves[j]
                slave_index  = seq_id_list.index(slave)
                slave__index = seq_id_list.index(slave_)
                distance_value = distance_matrix[slave_index][slave__index]
                tem_sum += distance_value
            if tem_sum < minimum_dis:
                minimum_dis = tem_sum
                minimum_index = slave
        new_centers.append(minimum_index)
    return new_centers


def assign_cluster(seq_id_list, centers, distance_matrix):

    assert len(centers) > 0, "Cluster Num > 0"
    assignments = {}
    for center in centers:
        assignments[center] = []
    for seq_id in seq_id_list:
        minimum_dis = 1000000
        minimum_index = 0
        for i in range(len(centers)):
            index_0 = seq_id_list.index(seq_id)
            index_1 = seq_id_list.index(centers[i])
            distance_value = distance_matrix[index_0][index_1]
            if distance_value < minimum_dis:
                minimum_dis = distance_value
                minimum_index = i
        tem_center = centers[minimum_index]
        assignments[tem_center].append(seq_id)
    return assignments


def random_generate_k_centers(k, seq_id_list):
    centers = []
    for i in range(k):
        centers.append(seq_id_list[randint(0, len(seq_id_list)-1)])
    return centers


def Kmediods(n_clusters, distance_matrix):
    
    assert n_clusters > 0, "k Num > 0"
    seq_id_list = [i for i in range(len(distance_matrix))]
    centers = random_generate_k_centers(n_clusters, seq_id_list)
    centers.sort()
    assignments = assign_cluster(seq_id_list, centers, distance_matrix)
    iteration_num = 0
    while True:
        new_centers = update_center(assignments, centers, distance_matrix, seq_id_list)
        new_centers.sort()
        if (new_centers == centers) or (iteration_num == 20):
            break
        centers = new_centers
        assignments = assign_cluster(seq_id_list, centers, distance_matrix)
        iteration_num += 1
    labels = [0] * len(seq_id_list)
    cluster_id = 0
    for center in assignments:
        for slave in assignments[center]:
            labels[slave] = cluster_id
        cluster_id += 1
    return labels,list(assignments.keys())


if __name__ == "__main__":
    pass
