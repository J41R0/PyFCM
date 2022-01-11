import math
import numpy as np
from numba import njit


@njit
def _compute_u(data, v, alpha, r1, r2):
    u = np.zeros((len(data), len(v)), dtype=np.float64)
    c = len(alpha)
    for i in range(u.shape[0]):
        denom = 0.0
        for t in range(c):
            d_it = np.linalg.norm(data[i] - v[t])
            denom += math.exp((-(d_it ** 2) + (r1 * math.log(alpha[t]))) / r2)
        for k in range(u.shape[1]):
            d_ik = np.linalg.norm(data[i] - v[k])
            u[i][k] = math.exp((-(d_ik ** 2) + (r1 * math.log(alpha[k]))) / r2) / denom
    return u


@njit
def _update_alpha(u, alpha, r1, r3):
    new_alpha = alpha.copy()
    n = len(u)
    c = len(alpha)
    ln_accum = np.sum(np.multiply(alpha, np.log2(alpha)))
    for k in range(c):
        par_content = np.log2(alpha[k]) - ln_accum
        pre_val = 0.0
        for i in range(len(u)):
            pre_val += u[i][k] + ((r3 / r1) * alpha[k] * par_content)
        new_alpha[k] = pre_val / n
    return new_alpha


@njit
def _update_r3(alpha_old, new_alpha, u):
    # TODO: add niu function
    niu = 1
    n = len(u)
    c = len(alpha_old)
    num = 0.0
    ku_sum = []
    v2_denom_accum = []

    alpha_old_sum = 0.0
    for t in range(c):
        alpha_old_sum += alpha_old[t] * math.log(alpha_old[t])
    for k in range(c):
        num += math.exp(-niu * n * abs(new_alpha[k] - alpha_old[k]))
        ku_sum.append(u.sum(0)[k] / n)
        v2_denom_accum.append(alpha_old[k] * alpha_old_sum)
    v1 = num / c
    v2 = (1 - max(ku_sum)) / (-max(v2_denom_accum))

    return min(v1, v2)


@njit
def _resize_references(alpha, u):
    j = 0
    n = len(u)
    c = alpha.size
    for t in range(c):
        if alpha[t] < (1 / n):
            j += 1
    new_c = c - j
    resized_alpha = np.empty(new_c, dtype=alpha.dtype)
    resized_u = np.zeros((n, new_c), dtype=u.dtype)
    j = 0
    for t in range(c):
        if alpha[t] >= 1 / n:
            resized_alpha[j] = alpha[t]
            for i in range(n):
                resized_u[i][j] = u[i][t]
            j += 1
    resized_alpha = resized_alpha / resized_alpha.sum()
    for i in range(n):
        resized_u[i] = resized_u[i] / resized_u[i].sum()

    return resized_alpha, resized_u


@njit
def _update_v(data, u, c):
    new_v = np.zeros((c, data.shape[1]), dtype=u.dtype)
    u_sum = u.sum(0)
    for k in range(c):
        num = np.zeros(data.shape[1], dtype=u.dtype)
        for i in range(len(data)):
            num = num + (data[i] * u[i][k])
        new_v[k] = num / u_sum[k]
    return new_v


@njit
def rl_fuzzy_cmeans(data, error=0.0005, max_iter=110):
    u = None
    c = data.shape[0]
    r1 = r2 = r3 = t = 1
    v = data.copy()
    alpha = np.full(c, 1 / c, dtype=np.float64)
    while t < max_iter:
        u = _compute_u(data, v, alpha, r1, r2)
        r1 = math.exp(-t / 10)
        r2 = math.exp(-t / 100)
        new_alpha = _update_alpha(u, alpha, r1, r3)
        r3 = _update_r3(alpha, new_alpha, u)
        new_alpha, u = _resize_references(new_alpha, u)
        c = len(new_alpha)
        if t >= 100 and (len(alpha) - len(new_alpha)) == 0:
            r3 = 0
        new_v = _update_v(data, u, c)
        if len(new_alpha) == 1 or len(new_v) == len(v) and np.linalg.norm(new_v - v) < error:
            v = new_v
            alpha = new_alpha
            break
        v = new_v
        alpha = new_alpha
        t += 1

    return v, u, alpha, t


class RlFuzzyCmeans:
    centroids = None

    def fit(self, X):
        v, u, alpha, t = rl_fuzzy_cmeans(X)
        print("Clusters: ", len(v))
        self.centroids = v

    def predict(self, X):
        y_pred = []
        for element in X:
            min_dist = np.linalg.norm(self.centroids[0] - element)
            res = 0
            for cent_pos in range(1, len(self.centroids)):
                curr_dist = np.linalg.norm(self.centroids[cent_pos] - element)
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    res = cent_pos
            y_pred.append(res)
        return y_pred
