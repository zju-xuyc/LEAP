
import pickle
import numpy as np

import config

from cluster.Kmediods import Kmediods
from cluster.Kmeans import Kmeans
from cluster.RangeCluster import RangeCluster
from cluster.MyCluster import MyCluster


from sklearn.cluster import KMeans as skKmeans
from sklearn.cluster import SpectralClustering as skSpectralClustering
from sklearn.cluster import AgglomerativeClustering as skAgglomerativeClustering
from sklearn.cluster import DBSCAN as skDBSCAN




def save_cluster_result(labels, write_path):
    with open(write_path, 'w') as f:
        for label in labels:
            f.write(str(label) + '\n')


def merge_cnned_embedding(root_read_path, write_path):
    b = np.load(root_read_path + "embedding_xb.npy")
    q = np.load(root_read_path + "embedding_xq.npy")
    t = np.load(root_read_path + "embedding_xt.npy")
    f = np.concatenate((t, q, b))
    pickle.dump(f, open(write_path, "wb"))
    

if __name__ == "__main__":

    clustering_mode_list   = config.clustering_mode_list
    clustering_method_list = config.clustering_method_list
    
    seq_length_path = config.seq_length_path
    seq_length_list = pickle.load(open(seq_length_path, "rb"))

    for clustering_mode in clustering_mode_list:
        if clustering_mode in ["rnn-l2", "rnn-cosine", "lstm-l2", "lstm-cosine", "gru-l2", "gru-cosine", "cnned", "transformer", "my-cnned"]:
            feature_path = config.feature_path_dict[clustering_mode]
            feature_list = pickle.load(open(feature_path, "rb"))
            for clustering_method in clustering_method_list:
                label_write_path = config.root_write_path + "/" + clustering_mode + "-" + clustering_method + "-" + "label.txt"
                labels = []
                print(label_write_path)

                distance_matrix = pickle.load(open(config.feature_distance_matrix_path_dict[clustering_mode], "rb"))
                if np.min(distance_matrix) < 0:
                    print("min distance:", np.min(distance_matrix))
                    distance_matrix += np.abs(np.min(distance_matrix))

                assert True not in np.isnan(distance_matrix), "distance matrix contains nan"
                assert True not in np.isinf(distance_matrix), "distance matrix contains inf"


                if clustering_method == "kmeans" and "cosine" in clustering_mode:
                    labels = Kmeans(config.parameter_dict["kmeans_k"], feature_list, clustering_mode)
                elif clustering_method == "kmeans":
                    cluster  = skKmeans(n_clusters = config.parameter_dict["kmeans_k"], random_state = 0).fit(feature_list)
                    labels   = cluster.labels_
                elif clustering_method == "spectral":
                    cluster  = skSpectralClustering(n_clusters = config.parameter_dict["spectral_n"], affinity = "precomputed").fit(4 - distance_matrix)
                    labels   = cluster.labels_
                elif clustering_method == "agglomerative":
                    cluster  = skAgglomerativeClustering(n_clusters = config.parameter_dict["agglomerative_n"], linkage='average', affinity = "precomputed").fit(distance_matrix)
                    labels   = cluster.labels_
                elif clustering_method == "dbscan":
                    cluster  = skDBSCAN(eps = config.parameter_dict["dbscan_eps"], min_samples = config.parameter_dict["dbscan_min_samples"], metric = "precomputed").fit(distance_matrix)
                    labels   = cluster.labels_
                elif clustering_method == "kmediods":
                    labels = Kmediods(config.parameter_dict["kmediods_k"], distance_matrix)
                elif clustering_method == "range":
                    labels = RangeCluster(seq_length_list, feature_list, config.parameter_dict["range_dis"], clustering_mode)
                elif clustering_method == "my-method":
                    labels = MyCluster(seq_length_list, feature_list, config.parameter_dict["my_range_dis"], clustering_mode)
                else:
                    print("Method ***{}*** has not been implemented.".format(label_write_path))
                save_cluster_result(labels, label_write_path)
        elif clustering_mode in ["traditional"]:
            similarity_matrix = pickle.load(open(config.similarity_matrix_path, "rb"))
            distance_matrix = 1 - similarity_matrix
        else:
            raise ValueError("Wrong clustering mode.")

