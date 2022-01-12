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
    # data = data.T
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
    n = data.shape[0]
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        u0 = np.random.rand(c, n)
        u0 = normalize_columns(u0)
        init = u0.copy()
    # print(init)
    u0 = init
    u = np.zeros((c, n), dtype=np.float64)
    for i in range(c):
        for j in range(n):
            u[i, j] = max(u0[i, j], np.finfo(np.float64).eps)

    # Initialize loop parameters
    jm = np.zeros(0)
    p = 0

    while p < maxiter - 1:
        u2 = u.copy()
        cntr, u, Jjm, d = _cmeans0(data, u2, m)
        jm = np.append(jm, Jjm)
        p += 1

        # Stopping rule
        if np.linalg.norm(u - u2) < error:
            break

    # error = np.linalg.norm(u - u2)
    fpc = _fp_coeff(u)
    return cntr, u, u0, d, jm, p, fpc


@njit
def fuzzy_cmeans(data: np.array, c: int, m=2.5, error=0.0001, maxiter=250, init=None, seed=None):
    """Simplify the clustering algorithm function call"""
    cntr, u, u0, d, jm, p, fpc = cmeans(data, c, m, error, maxiter, init, seed)
    return u, cntr
