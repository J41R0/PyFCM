import math
from math import exp
from py_fcm.__const import *


class Excitation:
    """
    All exitation functions must get a node as parameter
    """

    # node: node dict for all functions
    @staticmethod
    def kosko(node):
        """ TeX functions:
        not memory: A^{(t+1)}_i = f\left(\sum_{j=1}^N w_{ij}*A^{(t)}_j \right) , i \neq j
        use memory: A^{(t+1)}_i = f\left(A^{(t)}_i+\sum_{j=1}^N w_{ij}*A^{(t)}_j \right) , i \neq j
        """
        neighbors_val = node[NODE_AUX]
        node_val = node[NODE_VALUE]
        use_memory = node[NODE_USE_MEM]
        res = sum(neighbors_val)
        if use_memory:
            res += node_val
        return res

    @staticmethod
    def papageorgius(node):
        # to avoid saturation
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
    def get_by_name(func_name: str):
        """
        Get the function callable object from the function name
        Args:
            func_name: Excitation function name

        Returns: Function callable object if func_name is found, None otherwise

        """
        if func_name == "KOSKO":
            return Excitation.kosko
        if func_name == "PAPAGEORGIUS":
            return Excitation.papageorgius
        if func_name == "MEAN":
            return Excitation.mean
        return None

    @staticmethod
    def get_function_names() -> set:
        """
        Get available excitation function names
        Returns: Set of names

        """
        names = set()
        names.add("KOSKO")
        names.add("PAPAGEORGIUS")
        names.add("MEAN")
        return names


class Activation:
    @staticmethod
    def sigmoid(val, lambda_val=1):
        return 1.0 / (1.0 + exp(-1 * lambda_val * val))

    @staticmethod
    def sigmoid_lambda(x, y):
        res = -(math.log((1 / y) - 1) / x)
        return res

    @staticmethod
    def sigmoid_hip(val, lambda_val=2):
        # avoiding estimation errors
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
            if val < -1:
                return -1
            return val
        return 0

    @staticmethod
    def proportion(val, max_val, max_prop=1):
        prop = (max_prop * val) / max_val
        if prop > 1:
            return 1
        if prop < -1:
            return -1
        return prop

    @staticmethod
    def fuzzy_set(value, membership=[], val_list=[]):
        sign = 1.0
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

    @staticmethod
    def get_by_name(func_name: str):
        """
        Get the function callable object from the function name
        Args:
            func_name: Activation function name

        Returns: Function callable object if func_name is found, None otherwise

        """
        if func_name == "biestate":
            return Activation.bistate
        if func_name == "threestate":
            return Activation.tristate
        if func_name == "saturation":
            return Activation.saturation
        if func_name == "tan_hip":
            return Activation.sigmoid_hip
        if func_name == "sigmoid":
            return Activation.sigmoid
        if func_name == "sigmoid_hip":
            return Activation.sigmoid_hip
        if func_name == "sum_w":
            return Activation.sum_w
        if func_name == "proportion":
            return Activation.proportion
        return None

    @staticmethod
    def get_function_names() -> set:
        """
        Get available activation function names
        Returns: Set of names

        """
        names = set()
        names.add("biestate")
        names.add("threestate")
        names.add("saturation")
        names.add("tan_hip")
        names.add("sigmoid")
        names.add("sigmoid_hip")
        names.add("sum_w")
        names.add("proportion")
        return names


class Decision:
    @staticmethod
    def last(val_list: list) -> float:
        # return last value
        return val_list[-1]

    @staticmethod
    def mean(val_list: list) -> float:
        # return average execution value
        result = 0
        for elem in val_list:
            result += elem
        return result / len(val_list)

    @staticmethod
    def exited(val_list: list) -> float:
        # return highest execution value
        return max(val_list)

    @staticmethod
    def get_by_name(func_name: str):
        """
        Get the function callable object from the function name
        Args:
            func_name: Decision function name

        Returns: Function callable object if func_name is found, None otherwise

        """
        if func_name == "LAST":
            return Decision.last
        if func_name == "MEAN":
            return Decision.mean
        if func_name == "EXITED":
            return Decision.exited
        return None

    @staticmethod
    def get_function_names() -> set:
        """
        Get available excitation function names
        Returns: Set of names

        """
        names = set()
        names.add("LAST")
        names.add("MEAN")
        names.add("EXITED")
        return names


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
