import warnings

import numpy as np

from py_fcm.learning.discretization.clusters_estimation import estimate_clusters
from py_fcm.learning.discretization.rl_fuzzy_cmeans import rl_fuzzy_cmeans
from py_fcm.learning.discretization.fuzzy_cmeans import fuzzy_cmeans


def __create_three_clusters(val_list):
    if len(val_list) % 2 == 0:
        mid = int(len(val_list) / 2) - 1
        new_value = val_list[mid] + abs(val_list[mid] - val_list[mid + 1]) / 2
        new_list = np.concatenate((val_list[:mid + 1], [new_value], val_list[mid + 1:]))
        val_list = new_list.copy()
    else:
        new_list = val_list.copy()

    if new_list[0] <= 0:
        abs_min = 1 + abs(new_list[0])
        new_list = new_list + abs_min
    clusters_desc = np.zeros((len(new_list), 3))
    mid = int(len(new_list) / 2) + 1
    max = len(clusters_desc) - 1
    for pos in range(mid):
        clusters_desc[pos][2] = 0.0
        clusters_desc[pos][1] = pos / mid  # new_list[pos] / mid_val
        clusters_desc[pos][0] = 1.0 - clusters_desc[pos][1]

        clusters_desc[max - pos][0] = 0.0
        clusters_desc[max - pos][2] = clusters_desc[pos][0]
        clusters_desc[max - pos][1] = clusters_desc[pos][1]
    # start
    clusters_desc[0][0] = 1.0
    clusters_desc[0][1] = 0.0
    clusters_desc[0][2] = 0.0
    # mid
    clusters_desc[mid][0] = 0.0
    clusters_desc[mid][1] = 1.0
    clusters_desc[mid][2] = 0.0
    # end
    clusters_desc[-1][0] = 0.0
    clusters_desc[-1][1] = 0.0
    clusters_desc[-1][2] = 1.0
    clusters_desc = clusters_desc.T
    return val_list, clusters_desc


def fuzzy_feature_discretization(val_list, max_clusters=7, max_iter=200, seed=None,
                                 strategy="cmeans-gap", plot=False, att_name=None, plot_dir="."):
    """
      Estimate fuzzy clusters that define a continuous feature. Propose the amount of clusters to be used and return the
     membership degree of each provided point using fuzzy cmeans algorithm as kernel

    Args:
        val_list:
        max_clusters:
        gen_init_state:
        max_iter:
        seed:
        strategy: 'cmeans-gap' or 'cmeans-sil' for fuzzy cmeans or 'rl-cmeans' for robust learning fuzzy cmeans.
        force_clusters:
        plot:
        att_name:
        plot_dir:

    Returns: Membership dict of each provided value to each cluster. e.g. {'1.3':[0.01,0.57,0.23]}

    """
    # TODO: add a possiblistic fuzzy cmeans approach
    val_list = np.unique(val_list)
    if len(val_list) < 10:
        raise Exception("Too few (" + str(len(val_list)) + ") variations for fuzzy clustering on feature: " + att_name)
    val_list.sort()
    input_values = np.vstack(val_list)
    kwargs = {
        'maxiter': max_iter,
        'seed': seed
    }
    if strategy == 'cmeans-gap':
        num_clusters, clusters_desc = estimate_clusters(input_values, max_clusters, method='gap_concentration',
                                                        **kwargs)
    elif strategy == 'cmeans-sil':
        num_clusters, clusters_desc = estimate_clusters(input_values, max_clusters, method='fuzzy_silhouette',
                                                        **kwargs)
    elif strategy == 'cmeans-cfe':
        num_clusters, clusters_desc = estimate_clusters(input_values, max_clusters, method='combined_fuzzy_entropy',
                                                        **kwargs)
    elif strategy == 'rl-cmeans':
        centroids, clusters_desc, alpha, t = rl_fuzzy_cmeans(input_values, max_iter=max_iter)
        clusters_desc = clusters_desc.T
        num_clusters = len(alpha)
        if num_clusters > max_clusters:
            warnings.warn("Unexpected number of clusters found for " + att_name + ": " + str(num_clusters))
    else:
        raise Exception("Unsupported clustering method: " + strategy)
    # forcing clusters
    if num_clusters < 2:
        num_clusters = 3
        val_list, clusters_desc = __create_three_clusters(val_list)

    if plot:
        import matplotlib.pyplot as plt
        colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

        # plotting membership degree of values over found clusters
        for current in range(0, num_clusters):
            # plt.plot(val_list, clusters_desc[:, current], colors[current])
            plt.plot(range(0, len(val_list)), clusters_desc[current], colors[current])
        # show plotted values
        if att_name is not None:
            plt.savefig(plot_dir + '/' + att_name + '.png')
        else:
            plt.savefig(plot_dir + '/cluster_test.png')
        plt.close()

    return num_clusters, val_list, clusters_desc
