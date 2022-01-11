import warnings

import numpy as np
from numba import njit

from py_fcm.learning.utils import one_dimension_distance
from py_fcm.learning.discretization.fuzzy_cmeans import fuzzy_cmeans


@njit
def __estimate_change_points(values: np.ndarray, max_clusters: int):
    """
     Curve fit approach to find the amount of clusters to be used. Execute a curve fit process to input data and try to
    find the change points in input data, the amount of change points -1 is the number of clusters to be found and the
    change points can be used as algorithm initial state.
    Args:
        values:
        max_clusters:

    Returns:

    """
    n_samples = values.size

    min_interval = float(1 / max_clusters)
    mean = 0.0

    x_change_points = []
    y_change_points = []
    flag_grow = False
    for value in range(n_samples - 1):
        if abs(values[value] - values[value + 1]) > abs(mean) and not flag_grow:
            flag_grow = True
            x_change_points.append(value)
            y_change_points.append(values[value])

        if abs(values[value] - values[value + 1]) < abs(mean) and flag_grow:
            flag_grow = False
            x_change_points.append(value)
            y_change_points.append(values[value])

        if len(x_change_points) > 1 and value - x_change_points[-2] < n_samples * min_interval:
            del x_change_points[-1]
            del y_change_points[-1]

        mean = (abs(values[value] - values[value + 1]) + mean) / 2

    if len(x_change_points) > 1 and n_samples - 1 - x_change_points[-1] < n_samples * min_interval:
        del x_change_points[-1]
        del y_change_points[-1]
        x_change_points.append(n_samples - 1)
        y_change_points.append(values[-1])
    elif len(x_change_points) > 1:
        x_change_points.append(n_samples - 1)
        y_change_points.append(values[-1])

    return x_change_points, y_change_points


def gap_concentration(data: np.array, max_clusters: int):
    n_samples = data.size
    poly = np.polyfit(list(range(n_samples)), data[:, 0], max_clusters)
    func = np.poly1d(poly)
    image = func(np.linspace(0, data.size - 1, n_samples))
    change_points = __estimate_change_points(image, max_clusters)
    return image, change_points


def fuzzy_silhouette(data: np.array, max_clusters: int, alpha=2, function=fuzzy_cmeans, **kwargs):
    from sklearn.metrics import pairwise_distances_chunked
    if function is not None:
        dist_matrix = next(pairwise_distances_chunked(data))
        clusters = 0
        final_fuzzy_silhouette = 0
        final_desc = data
        for n_clusters in range(2, max_clusters + 1):
            clusters_desc, centroids = function(data, n_clusters, **kwargs)
            silhouette = np.zeros(len(data))
            weight = np.zeros(len(data))
            for i in range(len(data)):
                curr_membership = clusters_desc[:, i]
                if curr_membership[0] > curr_membership[1]:
                    max_val = curr_membership[0]
                    max_pos = 0
                    sec_max = curr_membership[1]
                    sec_max_pos = 1
                else:
                    max_val = curr_membership[1]
                    max_pos = 1
                    sec_max = curr_membership[0]
                    sec_max_pos = 0
                for j in range(2, n_clusters):
                    if curr_membership[j] > max_val:
                        sec_max = max_val
                        sec_max_pos = max_pos
                        max_val = curr_membership[j]
                        max_pos = j
                    elif sec_max < curr_membership[j] != max_val:
                        sec_max = curr_membership[j]
                        sec_max_pos = j
                weight[i] = (max_val - sec_max) ** alpha
                a = np.sum(np.multiply(dist_matrix[i], clusters_desc[max_pos])) / (len(data) - 1)
                b = np.sum(np.multiply(dist_matrix[i], clusters_desc[sec_max_pos])) / (len(data) - 1)
                silhouette[i] = (b - a) / max(b, a)
            fuzzy_silhouette = np.sum(np.multiply(weight, silhouette)) / np.sum(weight)
            if fuzzy_silhouette > final_fuzzy_silhouette:
                final_fuzzy_silhouette = fuzzy_silhouette
                clusters = n_clusters
                final_desc = clusters_desc
        return clusters, final_desc, final_fuzzy_silhouette


def __fuzzy_cross_entropy(u_i, u_j):
    return np.sum(
        u_i * np.log2(u_i / (u_i / 2 + u_j / 2)) +
        (1 - u_i) * np.log2((1 - u_i) / (1 - (u_i + u_j) / 2))
    )


def cfe(data: np.array, max_clusters: int, function=fuzzy_cmeans, **kwargs):
    if function is not None:
        n = len(data)
        clusters = 0
        final_cfe = 0
        final_desc = data
        absolute_center = np.sum(data) / len(data)
        for n_clusters in range(2, max_clusters + 1):
            clusters_desc, centroids = function(data, n_clusters, **kwargs)
            sum_hui = wsgd = bgds = sfce = 0.0
            for cluster_pos in range(n_clusters):
                curr_cluster = clusters_desc[cluster_pos]
                a_i = np.sum(curr_cluster)
                sum_val = np.sum(np.multiply(curr_cluster, np.log2(curr_cluster)) +
                                 np.multiply(1 - curr_cluster, np.log2(1 - curr_cluster)))
                h_u_i = -sum_val / (n * np.log2(2))
                sum_hui += h_u_i

                wsgd += np.sum(curr_cluster * (np.linalg.norm(data - centroids[cluster_pos]) ** 2))
                bgds += a_i * np.linalg.norm(centroids[cluster_pos] - absolute_center) ** 2
                for j in range(n_clusters):
                    if cluster_pos != j:
                        sfce += (__fuzzy_cross_entropy(clusters_desc[cluster_pos], clusters_desc[j]) +
                                 __fuzzy_cross_entropy(clusters_desc[j], clusters_desc[cluster_pos]))
            # final calculations
            sfce = (2 / (n_clusters * (n_clusters - 1))) * sfce
            fe = sum_hui / n_clusters
            ch = (bgds / (n_clusters - 1)) * ((n - n_clusters) / wsgd)
            mc = sfce / fe
            cfe = (mc + ch) / 2
            if cfe > final_cfe:
                final_cfe = cfe
                clusters = n_clusters
                final_desc = clusters_desc
        return clusters, final_desc, final_cfe


def estimate_clusters(data: np.array, max_clusters: int, method='gap_concentration', gen_init_state=True,
                      function=fuzzy_cmeans, **kwargs):
    available_methods = {'gap_concentration', 'fuzzy_silhouette', 'combined_fuzzy_entropy'}
    clusters_desc = None
    clusters = 1
    init_state = None
    kwargs['m'] = 2.5
    kwargs['error'] = 0.0001
    if max_clusters >= data.size:
        max_clusters = data.size / 2
    if method not in available_methods or method != 'gap_concentration' and function is None:
        warnings.warn("Unsupported clusters estimation function " + method + ". Used gap concentration instead.")
        method = gap_concentration
    if method == 'gap_concentration':
        image, change_points = gap_concentration(data, max_clusters)
        clusters = len(change_points[1]) - 1
        if gen_init_state:
            # generating initial state for cluster iterations
            init_list = np.reshape(data, (len(data), 1))
            # exclude last element if greater than 2
            if len(change_points[1]) > 2:
                init_cntrs = np.reshape(change_points[1][:-1], (len(change_points[1]) - 1, 1))
            else:
                init_cntrs = np.reshape(change_points[1], (len(change_points[1]), 1))
            init_state = one_dimension_distance(init_list, init_cntrs)
        if 'init' not in kwargs or kwargs['init'] is None:
            kwargs['init'] = init_state
        # clusters_desc, centrs = function(data, clusters, m=kwargs['m'], error=kwargs['error'],
        #                                  maxiter=kwargs['maxiter'], init=kwargs['init'], seed=kwargs['seed'])
        if clusters > 1:
            clusters_desc, centrs = function(data, clusters, **kwargs)
        else:
            clusters = 1
            clusters_desc = np.ones(len(data), dtype=np.float64)
    if method == 'fuzzy_silhouette':
        clusters, clusters_desc, final_fuzzy_silhouette = fuzzy_silhouette(data, max_clusters, alpha=2,
                                                                           function=function, **kwargs)
    if method == 'combined_fuzzy_entropy':
        clusters, clusters_desc, final_cfe = cfe(data, max_clusters, function=function, **kwargs)
    return clusters, clusters_desc
