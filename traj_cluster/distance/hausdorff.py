import numpy as np

def hausdorff(p, q):
    nA = p.shape[0]
    nB = q.shape[0]
    cmax = 0.
    for i in range(nA):
        cmin = np.inf
        for j in range(nB):
            d = np.linalg.norm(p[i] - q[j])
            if d<cmin:
                cmin = d
            if cmin<cmax:
                break
        if cmin>cmax and np.inf>cmin:
            cmax = cmin
    for j in range(nB):
        cmin = np.inf
        for i in range(nA):
            d = np.linalg.norm(p[i] - q[j])
            if d<cmin:
                cmin = d
            if cmin<cmax:
                break
        if cmin>cmax and np.inf>cmin:
            cmax = cmin
    return cmax