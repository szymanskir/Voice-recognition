import numpy as np

def dynamic_time_warping(x, y):
    """
    Measures the similarity between time series change
    :param x: first time series
    :param y: second time series
    :return: measure of similarity
    """

    # create local distance matrix
    distance_matrix = np.zeros((len(x), len(y)));
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            distance_matrix[i, j] = np.linalg.norm(x[i] - y[j])

    # initialize dtw matrix
    dtw_matrix = np.zeros((len(x), len(y)))
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
            dtw_matrix[i, j] = min(dtw_matrix[i-1, j-1], dtw_matrix[i-1, j], dtw_matrix[i, j-1]) + distance_matrix[i, j]

    return dtw_matrix[len(x)-1, len(y)-1]


def dist(x, y):
    return np.sqrt(np.sum((x-y)**2))
