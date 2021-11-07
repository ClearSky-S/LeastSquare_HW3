# Linear Algebra HW3
# 2021038131 장준혁(Jang JunHyuck)

import numpy as np


def inputMatrix():
    print('')
    print("Input Matrix")
    m = int(input('Dimension of column vector (m): '))
    n = int(input('Number of vectors (n): '))
    a = np.empty([m, n], dtype=float)  # make mxn empty matrix
    for j in range(0, n):
        while True:
            v = list(map(int, input('Column vector ' + str(j + 1) + ': ').split()))
            if len(v) is m:
                for i in range(m):
                    a[i][j] = v[i]
                break
            else:
                print("Error: dimension of vector should be " + str(m))
    print()
    print('Original Matrix')
    print(a)
    return a


def gs_orthogonalization(a):
    print('')
    print('Gram-Schmidt orthogonalization')
    m = a.shape[0]  # row size
    n = a.shape[1]  # column size
    q = np.empty([m, n], dtype=float)  # make mxn empty matrix

    for j in range(n):
        v = np.array(a[:, j])  # column vector j in original matrix
        u = v
        for i in range(j):
            ei = np.array(q[:, i])  # e i
            proj_ei_v = (np.dot(v, ei)) * ei
            u -= proj_ei_v
        e = u / np.linalg.norm(u)  # normalize vector
        for i in range(m):
            q[i][j] = e[i]
    return q


def find_linear_combination(q):
    print('')
    print('Find linear combination')
    m = a.shape[0]  # row size
    n = a.shape[1]  # column size
    v = list(map(int, input('please input vector whose dim is ' + str(m) + ': ').split()))
    c = np.empty([n], dtype=float)
    for i in range(n):
        c[i] = np.dot(q[:, i], v)
    print("coefficients: ", c)
    print("orthogonal matrix * coefficients = ", end='')
    print(q@c.transpose())


def check_orthonomality(q):
    print()
    print('Check orthonomality')
    m = a.shape[0]  # row size
    n = a.shape[1]  # column size
    for i in range(n-1):
        for j in range(i+1,n):
            print(f"check orthogonality of column {i} and {j}")
            print(f"column {i} * column {j} = " + str(np.dot(q[:, i],q[:, j])))


if __name__ == '__main__':
    a = []
    a = inputMatrix()

    q = gs_orthogonalization(a)
    print(q)

    check_orthonomality(q)

    find_linear_combination(q)
    find_linear_combination(q)
    find_linear_combination(q)

