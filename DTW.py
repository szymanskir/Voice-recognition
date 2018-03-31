import numpy as np
import scipy.io.wavfile


def dynamic_time_warping(x, y):
    """
    Measures the similarity between time series
    :param x: first time series
    :param y: second time series
    :return: measure of similarity
    """

    # create local distance matrix
    distance_matrix = np.zeros((len(x), len(y)), dtype=np.int16);
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            distance_matrix[i, j] = abs(x[i] - y[j])

    # initialize dtw matrix
    dtw_matrix = np.zeros((len(x), len(y)), dtype=np.int16)
    dtw_matrix[0, 0] = distance_matrix[0, 0]

    # first column initialization
    for i in range(1, len(x)):
        dtw_matrix[i, 0] = dtw_matrix[i-1, 0] + distance_matrix[i, 0]

    # first row initialization
    for j in range(1, len(y)):
        dtw_matrix[0, j] = dtw_matrix[0, j-1] + distance_matrix[0, j]

    # calculate remaining dtw_matrix elements
    for i in range(1, len(x)):
        for j in range(1, len(y)):
            dtw_matrix[i, j] = min(dtw_matrix[i-1, j-1], dtw_matrix[i-1, j], dtw_matrix[i, j-1])
            + distance_matrix[i, j]

    """
    pairs = [(len(x) - 1, len(y) - 1)]
    i = len(x) - 1
    j = len(y) - 1

    while i != 0 or j != 0:
        if i == 0:
            --j
            continue
        elif j == 0:
            --i
            continue

        smallest_cost = dtw_matrix[i-1, j-1]
        step_i = -1
        step_j = -1
        if smallest_cost > dtw_matrix[i-1, j]:
            smallest_cost = dtw_matrix[i-1, j]
            step_j = 0

        if smallest_cost > dtw_matrix[i, j-1]:
            step_i = 0
            step_j = -1

        i += step_i
        j += step_j
        last = pairs[0]

        if last[0] != i and last[1] != j:
            pairs.append((i, j))

    plt.plot(range(0, len(x)), x)
    plt.plot(range(0, len(y)), y)

    for p in pairs:
        plt.plot(p, (x[p[0]], y[p[1]]), 'r')

    plt.show()
    """
    return dtw_matrix[len(x)-1, len(y)-1]


def euclidean_distance(a, b):
    """
    Calculates the euclidian distance between a,b
    :param a: first number
    :param b: second number
    :return: distance between a and b
    """
    return np.abs(a - b)

