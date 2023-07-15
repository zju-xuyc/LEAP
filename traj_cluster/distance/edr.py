import numpy as np
import sys
"""
    how to use: python levenshtein.py str1 str2
    ex: python levenshtein.py saturday sunday 
"""

"""
 _                           _                  _                       
(_)_ __ ___  _ __ ___   __ _| |_ _   _ _ __ ___| |__   __ _  __ _ _ __  
| | '_ ` _ \| '_ ` _ \ / _` | __| | | | '__/ _ \ '_ \ / _` |/ _` | '_ \ 
| | | | | | | | | | | | (_| | |_| |_| | | |  __/ | | | (_| | (_| | | | |
|_|_| |_| |_|_| |_| |_|\__,_|\__|\__,_|_|  \___|_| |_|\__,_|\__,_|_| |_|
                                                                        
"""

def point_dis(a, b):
    dis = np.linalg.norm(a - b)
    return dis

def edr(seq1, seq2, eps = 0.002):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    #draw matrix
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y
    #compute
    for x in range(1, size_x):
        for y in range(1, size_y):
            if point_dis(seq1[x-1], seq2[y-1]) <= eps:
                edit = 0                    
            else:
                edit = 1
            x1y  = matrix[x-1,y] + 1         #[x - 1, y]
            x1y1 = matrix[x-1,y-1] + edit   #[x - 1, y - 1]
            xy1  = matrix[x,y-1] + 1         #[x, y - 1]
            matrix [x,y] = min(x1y, x1y1, xy1)
    
    return (matrix[size_x - 1, size_y - 1])


if __name__ == '__main__':
    str1 = str(sys.argv[1])
    str2 = str(sys.argv[2])
    print('Levenshtein distance: ', edr(str1,str2))