import numba
import numpy as np


@numba.njit()
def score_semiglobal(x, y, sub, gap):
    n, m = len(x), len(y)
    H = np.zeros((n + 1, m + 1), dtype=numba.int_)
    T = np.zeros((n + 1, m + 1), dtype=numba.int8)
    # H = np.zeros((n + 1, m + 1), dtype=int)
    # T = np.zeros((n + 1, m + 1), dtype=int)

    scores = np.zeros(3, dtype=numba.int_)
    # scores = np.zeros(3, dtype=int)

    for i in range(1, n + 1):
        H[i, 0] = H[i - 1, 0] - gap
        T[i, 0] = 1

    for j in range(1, m + 1):
        H[0, j] = 0
        T[0, j] = 2

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            scores[0] = H[i - 1, j - 1] + sub[x[i - 1], y[j - 1]]
            scores[1] = H[i - 1, j] - gap
            scores[2] = H[i, j - 1] - gap
            k = np.argmax(scores)
            H[i, j] = scores[k]
            T[i, j] = k

    return H, T


@numba.njit()
def backtrace(T, start, x, y, blank=-1):
    i, j = start
    path = []

    a = []
    b = []

    while T[i, j] != 3 and (i > 0 or j > 0):
        path.append((i, j))

        if T[i, j] == 0:  # substitution
            i -= 1
            j -= 1
            a.append(x[i])
            b.append(y[j])
        elif T[i, j] == 1:  # deletion
            i -= 1
            a.append(x[i])
            b.append(blank)
        elif T[i, j] == 2:  # insertion
            j -= 1
            a.append(blank)
            b.append(y[j])

    path.reverse()
    a.reverse()
    b.reverse()
    return path, a, b


@numba.njit()
def align_semiglobal(x, y, sub, gap):
    n, m = len(x), len(y)
    H, T = score_semiglobal(x, y, sub, gap)

    j = np.argmax(H[n, :])
    start = (n, j)

    path, a, b = backtrace(T, start, x, y)
    i, j = start

    for k in range(j, m):
        path.append((n, k + 1))
        a.append(-1)
        b.append(y[k])

    return path, a, b, H[i, j]
