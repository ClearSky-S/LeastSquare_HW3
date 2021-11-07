import numpy as np


def inputMatrix():
    print('')
    print("Input Matrix")
    m = int(input('Dimension of column vector (m): '))
    n = int(input('Number of vectors (n): '))
    a = np.empty([m, n], dtype=float)  # make mxn empty matrix
    for i in range(0, m):
        while True:
            v = list(map(int, input('Row ' + str(i + 1) + ': ').split()))
            if len(v) is n:
                a[i] = v
                break
            else:
                print("Error: Column number should be " + str(m))
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
    print(a)
    q = gs_orthogonalization(a)
    print(q)
    check_orthonomality(q)
    find_linear_combination(q)
