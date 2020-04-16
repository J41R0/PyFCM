import math
from math import exp
from py_fcm.__const import *


class Exitation:
    """
    All execution function must get a node as parameter
    """
    # node: node dict for all functions
    @staticmethod
    def kosko(node):
        # f(sum(wij * Ai) + Aj)
        neighbors_val = node[NODE_AUX]
        node_val = node[NODE_VALUE]
        use_memory = node[NODE_USE_MEM]
        res = sum(neighbors_val)
        if use_memory:
            res += node_val
        return res

    @staticmethod
    def papageorgius(node):
        neighbors_val = node[NODE_AUX]
        node_val = node[NODE_VALUE]
        use_memory = node[NODE_USE_MEM]
        res = sum(neighbors_val)
        if use_memory:
            res += (2 * node_val) - 1
        return res

    @staticmethod
    def mean(node):
        total = len(node[NODE_AUX])
        res = sum(node[NODE_AUX])
        if node[NODE_USE_MEM]:
            res += node[NODE_VALUE]
            total += 1
        if total == 0:
            return 0
        return res / total

    @staticmethod
    def actv_dim_event(node):
        no_zero_count = 0
        for val in node[NODE_AUX]:
            if val > 0:
                no_zero_count += 1
        if no_zero_count == len(node[NODE_AUX]):
            node[NODE_ACTIVE] = True
        return node[NODE_VALUE]


class Activation:
    @staticmethod
    def sigmoid1(val, amp=1):
        return 1.0 / (1.0 + exp((-1 * amp) * (val - 0.5)))

    @staticmethod
    def sigmoid(val, lambda_val=1):
        return 1.0 / (1.0 + exp(-1 * lambda_val * val))

    @staticmethod
    def sigmoid_lambda(x, y):
        res = -(math.log((1 / y) - 1) / x)
        return res

    @staticmethod
    def sigmoid_hip(val, lambda_val=2):
        # avoiding errors
        if (-1 * lambda_val * val) > 500:
            return (1.0 - exp(500)) / (1.0 + exp(500))
        else:
            return (1.0 - exp(-1 * lambda_val * val)) / (1.0 + exp(-1 * lambda_val * val))

    @staticmethod
    def sigmoid_hip_lambda(x, y):
        res = -(math.log((1 - y) / (1 + y)) / x)
        return res

    @staticmethod
    def saturation(val):
        if val < 0:
            return 0.0
        elif val > 1:
            return 1.0
        else:
            return val

    @staticmethod
    def bistate(val):
        if val <= 0.0:
            return 0.0
        return 1.0

    @staticmethod
    def tristate(val):
        if val <= 1.0 / 3.0:
            return 0.0
        elif val <= 2.0 / 3.0:
            return 0.5
        return 1.0

    @staticmethod
    def sum_w(val, weight):
        if val >= weight:
            if val > 1:
                return 1
            return val
        return 0

    @staticmethod
    def proportion(val, max_val, max_prop=1):
        prop = (max_prop * val) / max_val
        if prop > 1:
            return 1
        return prop

    @staticmethod
    def fuzzy_set(value, membership=[], val_list=[]):
        if value != 0:
            sign = value / abs(value)
        value = abs(value)
        # cmp values
        prev_value = 1
        next_value = -1

        # result positions
        prev_pos = 0
        next_pos = 0

        # calc result
        for elem_pos in range(0, len(val_list)):
            if value == val_list[elem_pos]:
                return membership[elem_pos]

            diff = float(value) - float(val_list[elem_pos])
            if diff > 0:
                if diff < prev_value:
                    prev_value = diff
                    prev_pos = elem_pos
            else:
                if diff > next_value:
                    next_value = diff
                    next_pos = elem_pos
        # minimum value
        if prev_value == 1:
            return min(membership)
        # maximum value
        if next_value == -1:
            return max(membership)
        # f(Xi) = (f(Xi-1)*Xi/Xi-1)*Xi-1_Xi_coef + (f(Xi+1)*Xi/Xi+1)*Xi+1_Xi_coef
        # inf_estimation = (membership[prev_pos] * value) / float(val_list[prev_pos])
        # sup_estimation = (membership[next_pos] * value) / float(val_list[next_pos])
        inf_estimation = membership[prev_pos]
        sup_estimation = membership[next_pos]
        diff = float(val_list[next_pos]) - float(val_list[prev_pos])
        if diff == 0:
            return membership[next_pos]
        # calc influence coefficents
        inf_coef = 1 - ((value - float(val_list[prev_pos])) / diff)
        # 1 - inf_coef
        sup_coef = 1 - ((float(val_list[next_pos]) - value) / diff)
        # result estimation according to distance between extremes

        estimation = sign * ((inf_coef * inf_estimation) + (sup_coef * sup_estimation))
        if estimation > 1:
            estimation = 1
        if estimation < -1:
            estimation = -1
        return estimation


class Fuzzy:
    @staticmethod
    def defuzzyfication(memb_val, min_scale, norm_val, membership=[], val_list=[]):
        """
        Estimate possibles neron outputs according to activation.
        Args:
            memb_val: activation value
            min_scale: minimum value in values list for scale data
            norm_val: maximum value of scaled data for normalization
            membership: membership degree
            val_list: result values associated to membership

        Returns: List of possibles results

        """
        raise NotImplementedError("Not implemented function")

    @staticmethod
    def fuzzyfication(value, min_scale, norm_val, membership=[], val_list=[]):
        """
        Estimate the neuron activation according to described discrete fuzzy set.
        Args:
            value: value for estimate activation
            min_scale: minimum value in values list for scale data
            norm_val: maximum value of scaled data for normalization
            membership: membership degree list, each value must belong to domain [-1,1]
            val_list: result values associated to membership and len(val_list) = len(membership)

        Returns: Estimated activation

        """
        value = (float(value) + min_scale) / norm_val

        # cmp values
        prev_value = 1
        next_value = -1

        # result positions
        prev_pos = 0
        next_pos = 0

        # calc result
        for elem_pos in range(0, len(val_list)):
            if value == val_list[elem_pos]:
                return membership[elem_pos]

            diff = float(value) - float(val_list[elem_pos])
            if diff > 0:
                if diff < prev_value:
                    prev_value = diff
                    prev_pos = elem_pos
            else:
                if diff > next_value:
                    next_value = diff
                    next_pos = elem_pos
        # minimum value
        if prev_value == 1:
            return min(membership)
        # maximum value
        if next_value == -1:
            return max(membership)
        # estimate value
        inf_estimation = membership[prev_pos]
        sup_estimation = membership[next_pos]
        diff = float(val_list[next_pos]) - float(val_list[prev_pos])
        if diff == 0:
            return membership[next_pos]
        # calc influence coefficents
        inf_coef = 1 - ((value - float(val_list[prev_pos])) / diff)
        # 1 - inf_coef
        sup_coef = 1 - ((float(val_list[next_pos]) - value) / diff)
        # result estimation according to distance between extremes
        estimation = (inf_coef * inf_estimation) + (sup_coef * sup_estimation)
        if estimation > 1:
            estimation = 1
        if estimation < -1:
            estimation = -1
        return estimation


class Relation:
    """
    All execution function must have the same params described in supp function, even if not used
    """

    # support
    @staticmethod
    def supp(p_q, p_nq, np_q, np_nq):
        return p_q / (p_q + p_nq + np_q + np_nq)

    # confidence
    @staticmethod
    def conf(p_q, p_nq, np_q, np_nq):
        if (p_q + p_nq) != 0:
            return p_q / (p_q + p_nq)
        else:
            return p_q

    # lift
    @staticmethod
    def lift(p_q, p_nq, np_q, np_nq):
        return (p_q / (p_q + p_nq)) / ((p_q + np_q) / (p_q + p_nq + np_q + np_nq))

    # red odss ratio
    @staticmethod
    def rodr(p_q, p_nq, np_q, np_nq):
        if (p_nq + np_q) != 0:
            return (p_q + np_nq) / (p_nq + np_q)
        else:
            # zero div behavior
            return p_q + np_nq

    # odss ratio
    @staticmethod
    def odr(p_q, p_nq, np_q, np_nq):
        if (p_nq * np_q) != 0:
            return (p_q * np_nq) / (p_nq * np_q)
        else:
            # zero div behavior
            return p_q * np_nq

    # positive influence
    @staticmethod
    def pos_inf(p_q, p_nq, np_q, np_nq):
        total = p_q + p_nq + np_q + np_nq
        if (p_nq + np_q) != 0:
            pos_inf = (p_q / (p_nq + np_q)) / (p_q + p_nq)
        else:
            # zero div behavior
            pos_inf = p_q / (p_q + p_nq)

        if (p_q + np_nq) != 0:
            neg_inf = (p_nq / (p_q + np_nq)) / (p_q + p_nq)
        else:
            # zero div behavior
            neg_inf = p_nq / (p_q + p_nq)
        if pos_inf > neg_inf:
            return pos_inf
        if pos_inf < neg_inf:
            return -1 * neg_inf
        return 0

    @staticmethod
    def simple(p_q, p_nq, np_q, np_nq):
        total = p_q + p_nq + np_q + np_nq
        if p_q == p_nq:
            return 0
        if p_q - p_nq != 0:
            return ((p_q - p_nq) + (np_nq + np_q)) / total
        else:
            # zero div behavior
            return ((p_q - p_nq) - (np_nq + np_q)) / total