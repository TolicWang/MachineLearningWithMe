# @Time    : 2018/11/22 9:01
# @Email  : wangchengo@126.com
# @File   : accFscore.py
# package info:
#               python 3.6
#               sklearn 0.20.0
#               numpy 1.15.2

from sklearn.metrics import accuracy_score


def get_acc_fscore(y, y_pre):
    import numpy as np

    n = len(y_pre)
    p = np.unique(y)
    c = np.unique(y_pre)
    p_size = len(p)
    c_size = len(c)

    a = np.ones((p_size, 1), dtype=int) * y  # p_size by 1  *  1 by n   ==> p_size by n
    b = p.reshape(p_size, 1) * np.ones((1, n), dtype=int)  # p_size by 1 * 1 by n ==> p_size by n
    pid = (a == b) * 1  # p_size by n

    a = np.ones((c_size, 1), dtype=int) * y_pre  # c_size by 1 * 1 by n ==> c_size by n
    b = c.reshape(c_size, 1) * np.ones((1, n))
    cid = (a == b) * 1  # c_size by n

    cp = np.dot(cid, pid.T)
    pj = np.sum(cp, axis=0)
    ci = np.sum(cp, axis=1)

    precision = cp / (ci.reshape(len(ci), 1) * np.ones((1, p_size), dtype=float))
    recall = cp / (np.ones((c_size, 1), dtype=float) * pj.reshape(1, len(pj)))

    F = (2 * precision * recall) / (precision + recall)

    F = np.nan_to_num(F)

    temp = (pj / float(pj.sum())) * np.max(F, axis=0)
    Fscore = np.sum(temp, axis=0)

    temp = np.max(cp, axis=1)
    Accuracy = np.sum(temp, axis=0) / float(n)

    return (Fscore, Accuracy)


if __name__ == "__main__":
    y1 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    y2 = [0, 0, 0, 0, 0, 1, 1, 1, 0, 1]
    print(accuracy_score(y1, y2))
    print(get_acc_fscore(y1, y2))
