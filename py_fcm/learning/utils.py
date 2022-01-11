import numpy as np
from numba import njit


# TODO: use njit decorator
def gen_discrete_feature_matrix(val_list, unique_values):
    val_num = len(val_list)
    feat_num = len(unique_values)
    matrix = np.zeros((feat_num, val_num), dtype=np.int8)
    for val_pos in range(val_num):
        for single_val_pos in range(feat_num):
            if val_list[val_pos] == unique_values[single_val_pos]:
                matrix[single_val_pos][val_pos] = 1
    return matrix


@njit
def calc_concepts_coefficient(array1: np.array, array2: np.array):
    if len(array1) == len(array2):
        a_b = 0
        na_b = 0
        a_nb = 0
        na_nb = 0
        for val_pos in range(len(array1)):
            a_b += array1[val_pos] * array2[val_pos]
            a_nb += array1[val_pos] * (1 - array2[val_pos])
            na_b += (1 - array1[val_pos]) * array2[val_pos]
            na_nb += (1 - array1[val_pos]) * (1 - array2[val_pos])
        # return: a->b relation coefficients, b->a relation coefficients
        return (a_b, a_nb, na_b, na_nb), (a_b, na_b, a_nb, na_nb)


@njit
def one_dimension_distance(data: np.ndarray, centers: np.ndarray):
    res = np.empty((len(centers), len(data)), dtype=np.float64)
    for c_pos in range(len(centers)):
        for d_pos in range(len(data)):
            res[c_pos][d_pos] = np.linalg.norm(data[d_pos] - centers[c_pos])
    return res


@njit
def normalize_columns(columns):
    """
    Normalize columns of matrix.
    Parameters
    ----------
    columns : 2d array (M x N)
        Matrix with columns
    Returns
    -------
    normalized_columns : 2d array (M x N)
        columns/np.sum(columns, axis=0, keepdims=1)
    """

    # broadcast sum over columns
    normalized_columns = columns / np.sum(columns, axis=0)

    return normalized_columns


@njit
def normalize_power_columns(columns: np.ndarray, exponent: float):
    """
    Calculate normalize_columns(x**exponent)
    in a numerically safe manner.
    Parameters
    ----------
    columns : 2d array (M x N)
        Matrix with columns
    exponent : float
        Exponent
    Returns
    -------
    result : 2d array (M x N)
        normalize_columns(x**exponent) but safe
    """

    assert np.all(columns >= 0.0)

    columns = columns.astype(np.float64)

    # values in range [0, 1]
    # columns = columns / np.max(columns, axis=0)
    for col in range(columns.shape[1]):
        columns[:, col] /= np.max(columns[:, col])

    # values in range [eps, 1]

    columns = np.fmax(columns, np.finfo(columns.dtype).eps)

    if exponent < 0:
        # values in range [1, 1/eps]
        # columns /= np.min(columns, axis=0)
        for col in range(columns.shape[1]):
            columns[:, col] /= np.min(columns[:, col])

        # values in range [1, (1/eps)**exponent] where exponent < 0
        # this line might trigger an underflow warning
        # if (1/eps)**exponent becomes zero, but that's ok
        columns = columns ** exponent
    else:
        # values in range [eps**exponent, 1] where exponent >= 0
        columns = columns ** exponent

    result = normalize_columns(columns)

    return result


# ensure compilation
# gen_discrete_feature_matrix(np.array(['1', '1', '2', '2']), np.array(['1', '2']))
calc_concepts_coefficient(np.array([1, 0, 0, 1]), np.array([0, 0, 1, 1]))
one_dimension_distance(np.array([[-1], [-1.5], [-2]]), np.array([[0.1], [0.3]]))
normalize_columns(np.array([[1.0, 1.8], [0.3, 0.7]]))
normalize_power_columns(np.array([[1.0, 1.8], [0.3, 0.7]]), 2)
