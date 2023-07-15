import numpy as np

from sklearn.cluster import KMeans

def knn(similarity_matrix, k, sigma = 1.0):
    length = len(similarity_matrix)
    adj = np.zeros((length, length))

    for i in range(length):
        dist_with_index = zip(similarity_matrix[i], range(length))
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
        neighbours_id = [dist_with_index[m][1] for m in range(k+1)] # xi's k nearest neighbours

        for j in neighbours_id: # xj is xi's neighbour
            adj[i][j] = np.exp(-similarity_matrix[i][j]/2/sigma/sigma)
            adj[j][i] = adj[i][j] # mutually

    return adj


def cal_laplacian_matrix(adjacent_matrix):
    
    # compute the Degree Matrix: D=sum(A)
    degree_matrix = np.sum(adjacent_matrix, axis=1)

    # compute the Laplacian Matrix: L=D-A
    laplacian_matrix = np.diag(degree_matrix) - adjacent_matrix


    # normailze
    # D^(-1/2) L D^(-1/2)
    sqrt_degree_matrix = np.diag(1.0 / (degree_matrix ** (0.5)))
    return np.dot(np.dot(sqrt_degree_matrix, laplacian_matrix), sqrt_degree_matrix)




def SpectralClustering(k2, distance_matrix, k1 = 100):
    
    similarity_matrix = 1 - distance_matrix

    Adjacent = knn(similarity_matrix, k = k1)

    Laplacian = cal_laplacian_matrix(Adjacent)
    
    x, V = np.linalg.eig(Laplacian)

    x = x.real
    V = V.real

    x = zip(x, range(len(x)))
    x = sorted(x, key = lambda x:x[0])

    H = np.vstack([V[:,i] for (v, i) in x[:500]]).T

    sp_kmeans = KMeans(n_clusters = k2).fit(H)
    return sp_kmeans.labels_


if __name__ == "__main__":
    pass
    
    