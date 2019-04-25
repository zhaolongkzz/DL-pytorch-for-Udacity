import numpy as np

# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    cross_entropy = 0
    for y, p in zip(Y, P):
        cross_entropy += -(y * np.log(p) + (1-y) * np.log(1-p))
    return cross_entropy

def cross_entropy_solution(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    print(type(P))
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))

if __name__ == '__main__':
    Y= [1, 0, 1, 1]
    P= [0.4, 0.6, 0.1, 0.5]

    zipped = zip(Y,P)         # 打包为元组的列表
    # [(1, 0.4), (0, 0.6), (1, 0..1), (1, 0.5)]
    A = zip(*zipped)          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
    # [(1, 0, 1, 1), (0.4, 0.6, 0.1, 0.5)]

    cross_entropy = cross_entropy(Y, P)
    print(cross_entropy)

    cross_entropy_solution = cross_entropy_solution(Y, P)
    print(cross_entropy_solution)
