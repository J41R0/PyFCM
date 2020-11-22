import numpy as np
from numba import njit

from py_fcm.learning.utils import normalize_columns, normalize_power_columns, one_dimension_distance


@njit
def _cmeans0(data, u_old, m):
    """
    Single step in generic fuzzy c-means clustering algorithm.
    Modified from Ross, Fuzzy Logic w/Engineering Applications (2010),
    pages 352-353, equations 10.28 - 10.35.
    Parameters inherited from cmeans()
    """
    # Normalizing, then eliminating any potential zero values.
    u_old = normalize_columns(u_old)

    u_old = np.fmax(u_old, np.finfo(np.float64).eps)

    um = u_old ** m

    # Calculate cluster centers
    data = data.T
    cntr = um.dot(data) / np.atleast_2d(um.sum(axis=1)).T

    d = one_dimension_distance(data, cntr)
    d = np.fmax(d, np.finfo(np.float64).eps)

    jm = (um * d ** 2).sum()

    u = normalize_power_columns(d, - 2. / (m - 1))

    return cntr, u, jm, d


@njit
def _fp_coeff(u):
    """
    Fuzzy partition coefficient `fpc` relative to fuzzy c-partitioned
    matrix `u`. Measures 'fuzziness' in partitioned clustering.
    Parameters
    ----------
    u : 2d array (C, N)
        Fuzzy c-partitioned matrix; N = number of data points and C = number
        of clusters.
    Returns
    -------
    fpc : float
        Fuzzy partition coefficient.
    """
    n = u.shape[1]

    return np.trace(u.dot(u.T)) / float(n)


@njit
def cmeans(data, c, m, error, maxiter,
           init=None, seed=None):
    """
    Fuzzy c-means clustering algorithm [1].
    Parameters
    ----------
    data : 2d array, size (S, N)
        Data to be clustered.  N is the number of data sets; S is the number
        of features within each sample vector.
    c : int
        Desired number of clusters or classes.
    m : float
        Array exponentiation applied to the membership function u_old at each
        iteration, where U_new = u_old ** m.
    error : float
        Stopping criterion; stop early if the norm of (u[p] - u[p-1]) < error.
    maxiter : int
        Maximum number of iterations allowed.
    metric: string
        By default is set to euclidean. Passes any option accepted by
        ``scipy.spatial.distance.cdist``.
    init : 2d array, size (c, N)
        Initial fuzzy c-partitioned matrix. If none provided, algorithm is
        randomly initialized.
    seed : int
        If provided, sets random seed of init. No effect if init is
        provided. Mainly for debug/testing purposes.
    Returns
    -------
    cntr : 2d array, size (c, S)
        Cluster centers.  Data for each center along each feature provided
        for every cluster (of the `c` requested clusters).
    u : 2d array, (c, N)
        Final fuzzy c-partitioned matrix.
    u0 : 2d array, (c, N)
        Initial guess at fuzzy c-partitioned matrix (either provided init or
        random guess used if init was not provided).
    d : 2d array, (c, N)
        Final Euclidian distance matrix.
    jm : 1d array, length P
        Objective function history.
    p : int
        Number of iterations run.
    fpc : float
        Final fuzzy partition coefficient.
    Notes
    -----
    The algorithm implemented is from Ross et al. [1]_.
    Fuzzy C-Means has a known problem with high dimensionality datasets, where
    the majority of cluster centers are pulled into the overall center of
    gravity. If you are clustering data with very high dimensionality and
    encounter this issue, another clustering method may be required. For more
    information and the theory behind this, see Winkler et al. [2]_.
    References
    ----------
    .. [1] Ross, Timothy J. Fuzzy Logic With Engineering Applications, 3rd ed.
           Wiley. 2010. ISBN 978-0-470-74376-8 pp 352-353, eq 10.28 - 10.35.
    .. [2] Winkler, R., Klawonn, F., & Kruse, R. Fuzzy c-means in high
           dimensional spaces. 2012. Contemporary Theory and Pragmatic
           Approaches in Fuzzy Computing Utilization, 1.
    """
    # Setup u0
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        n = data.shape[1]
        u0 = np.random.rand(c, n)
        u0 = normalize_columns(u0)
        init = u0.copy()
    u0 = init
    u = np.fmax(u0, np.finfo(np.float64).eps)

    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0

    # Main cmeans loop
    while p < maxiter - 1:
        u2 = u.copy()
        cntr, u, Jjm, d = _cmeans0(data, u2, m)
        # jm = np.hstack((jm, Jjm))
        jm = np.append(jm, Jjm)
        p += 1

        # Stopping rule
        if np.linalg.norm(u - u2) < error:
            break

    # Final calculations
    # error = np.linalg.norm(u - u2)
    fpc = _fp_coeff(u)

    return cntr, u, u0, d, jm, p, fpc


@njit
def __estimate_cluster_number(values: np.ndarray, max_clusters: int):
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


def __def_num_clusters(data: np.array, max_clusters):
    values = np.unique(data)
    n_samples = values.size
    values.sort()
    poly = np.polyfit(list(range(n_samples)), values, max_clusters)
    func = np.poly1d(poly)
    image = func(np.linspace(0, data.size - 1, n_samples))
    if max_clusters >= data.size:
        max_clusters = data.size - 1
    change_points = __estimate_cluster_number(image, max_clusters)

    return values, image, change_points


# @njit
def __define_clusters(val_float_list, num_clusters, change_points, gen_init_state, strategy, max_iter, seed):
    # preparing data for clustering
    copy = np.vstack([val_float_list])
    init_state = None
    if gen_init_state:
        # generating initial state for cluster iterations
        init_list = np.reshape(val_float_list, (len(val_float_list), 1))
        # exclude last element
        init_cntrs = np.reshape(change_points[1][:-1], (len(change_points[1]) - 1, 1))
        # init_state = cdist(init_list, init_cntrs).T
        init_state = one_dimension_distance(init_list, init_cntrs)

    # execute clustering process
    # TODO: parametrize all data and use fpc clusters quality measure
    cntr, u, u0, d, jm, p, fpc = cmeans(copy, num_clusters, 2.5, error=0.0005, init=init_state,
                                        maxiter=max_iter, seed=seed)

    return u, cntr


def fuzzy_feature_discretization(val_list, max_clusters=7, gen_init_state=True, max_iter=500, seed=None,
                                 strategy="fuzzy", plot=False, att_name=None, plot_dir="."):
    """
      Estimate fuzzy clusters that define a continuous feature. Propose the amount of clusters to be used and return the
     membership degree of each provided point using fuzzy cmeans algorithm as kernel

    Args:
        val_list:
        max_clusters:
        gen_init_state:
        max_iter:
        seed:
        strategy:
        plot:
        att_name:
        plot_dir:

    Returns: Membership dict of each provided value to each cluster. e.g. {'1.3':[0.01,0.57,0.23]}

    """
    # TODO: add a possiblistic approach
    sorted_input_values, val_curve_fit, change_points = __def_num_clusters(val_list, max_clusters)
    num_clusters = len(change_points[0]) - 1
    if num_clusters > 1:
        clusters_description, cntr = __define_clusters(sorted_input_values, num_clusters, change_points, gen_init_state,
                                                       strategy, max_iter, seed)
    else:
        max_val = sorted_input_values[-1]
        min_val = sorted_input_values[0]
        clusters_description = np.zeros((1, sorted_input_values.size), dtype=np.float64)
        for pos in range(sorted_input_values.size):
            if min_val >= 0:
                clusters_description[0][pos] = sorted_input_values[pos] / max_val
            else:
                clusters_description[0][pos] = (sorted_input_values[pos] + abs(min_val)) / (max_val + abs(min_val))
        cntr = (max_val - abs(min_val)) / 2
        num_clusters = 1

    if plot:
        # TODO: parametrize plot clusters and estimations
        import matplotlib.pyplot as plt

        n_elem = len(sorted_input_values)
        if num_clusters > 1:
            # distributing found centroids over data array, set in nearest value
            cntr_pos = []
            for current in range(0, len(cntr)):
                diff_pos = 0
                diff = abs(sorted_input_values[0] - cntr[current][0])
                for value in range(0, n_elem):
                    if diff > abs(sorted_input_values[value] - cntr[current][0]):
                        diff = abs(sorted_input_values[value] - cntr[current][0])
                        diff_pos = value
                cntr_pos.append(diff_pos)
        else:
            cntr_pos = len(sorted_input_values) / 2
        # plotting original ordered data
        # for visualization
        plt.subplot(211)
        plt.plot(range(0, n_elem), sorted_input_values, 'b--')
        # plotting fitted data
        plt.plot(range(0, n_elem), val_curve_fit, 'r-')
        # plotting found change points
        plt.plot(change_points[0], change_points[1], 'go')
        # plotting found centroids
        plt.plot(cntr_pos, cntr, 'rs')

        plt.subplot(212)
        colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

        # plotting membership degree of values over found clusters
        for current in range(0, len(clusters_description)):
            plt.plot(range(0, n_elem), clusters_description[current], colors[current])
            pass
        # show plotted values
        if att_name is not None:
            plt.savefig(plot_dir + '/' + att_name + '.png')
        plt.close()

    return num_clusters, sorted_input_values, clusters_description


# ensure compilation
fuzzy_feature_discretization(np.random.normal(0, 0.11, 8), att_name='test', plot=True, max_iter=2)
