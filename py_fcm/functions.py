import math
from math import exp

import numpy as np
from numba.typed import List
from numba import njit

from py_fcm.__const import *


# vectorized inference process
@njit
def vectorized_run(state_vector: np.ndarray, relation_matrix: np.ndarray, functions: np.ndarray, func_args: List,
                   reduce_values: np.ndarray, memory_usage: List, avoid_saturation: List, max_iterations: int,
                   min_diff: float, extra_steps: int):
    output = np.full((state_vector.size, max_iterations), 2.0)
    keep_execution = True
    extra_steps_counter = extra_steps
    it_counter = max_iterations

    for val_pos in range(state_vector.size):
        output[val_pos][0] = state_vector[val_pos]
        if avoid_saturation[val_pos]:
            state_vector[val_pos] = (2 * state_vector[val_pos]) - 1

    current_step = 1
    difference = min_diff
    while keep_execution:
        it_counter = it_counter - 1
        if it_counter <= 0:
            keep_execution = False
        new_state = np.dot(state_vector, relation_matrix)
        for val_pos in range(state_vector.size):
            if memory_usage[val_pos]:
                new_state[val_pos] = new_state[val_pos] + state_vector[val_pos]
            if reduce_values[val_pos] > 0:
                new_state[val_pos] = new_state[val_pos] / reduce_values[val_pos]
            new_state[val_pos] = exec_actv_function(functions[val_pos], new_state[val_pos], func_args[val_pos])
            output[val_pos][current_step] = new_state[val_pos]

            if avoid_saturation[val_pos]:
                new_state[val_pos] = 2 * new_state[val_pos] - 1

        state_vector = new_state
        current_step = current_step + 1

        if current_step > 1:
            difference = abs(np.sum(output[:, current_step - 1]) - np.sum(output[:, current_step - 2]))
        if difference < min_diff:
            extra_steps_counter = extra_steps_counter - 1
            if extra_steps_counter == 0:
                keep_execution = False
        else:
            extra_steps_counter = extra_steps
    return output


# activation functions relations

@njit
def sigmoid(val: float, lambda_val=1.0) -> float:
    return 1.0 / (1.0 + exp(-1 * lambda_val * val))


@njit
def sigmoid_lambda(x: float, y: float) -> float:
    res = -(math.log((1 / y) - 1) / x)
    return res


@njit
def sigmoid_hip(val: float, lambda_val=2.0) -> float:
    # avoiding estimation errors
    if (-1 * lambda_val * val) > 500:
        return (1.0 - exp(500)) / (1.0 + exp(500))
    else:
        return (1.0 - exp(-1 * lambda_val * val)) / (1.0 + exp(-1 * lambda_val * val))


@njit
def sigmoid_hip_lambda(x: float, y: float) -> float:
    res = -(math.log((1 - y) / (1 + y)) / x)
    return res


@njit
def saturation(val: float) -> float:
    if val < 0:
        return 0.0
    elif val > 1:
        return 1.0
    else:
        return val


@njit
def bistate(val: float) -> float:
    if val <= 0.0:
        return 0.0
    return 1.0


@njit
def threestate(val: float) -> float:
    if val <= 1.0 / 3.0:
        return 0.0
    elif val <= 2.0 / 3.0:
        return 0.5
    return 1.0


@njit
def greater_cond_equality(val: float, weight=-1.0) -> float:
    if val >= weight:
        if val > 1:
            return 1
        if val < -1:
            return -1
        return val
    return 0


@njit
def lower_cond_equality(val: float, weight=1.0) -> float:
    if val <= weight:
        if val > 1:
            return 1
        if val < -1:
            return -1
        return val
    return 0


@njit
def fuzzy_set(value: float, membership=np.empty(1, dtype=np.float64),
              val_list=np.empty(1, dtype=np.float64)) -> float:
    # is assumed that the list of values (val_list) is sorted from lowest to gratest

    negative_activation = False
    if 0.0 <= val_list.min() <= 1.0 and 0.0 <= val_list.max() <= 1.0 and value < 0.0:
        negative_activation = True
        value = abs(value)

    # result positions
    prev_pos = 0

    # find nearest values index
    index = (np.abs(val_list - value)).argmin()
    if val_list[index] == value:
        if not negative_activation:
            return membership[index]
        else:
            return -1 * (1 - membership[index])
    if index == 0:
        if val_list[index] > value:
            if not negative_activation:
                return membership[index]
            else:
                return -1 * (1 - membership[index])
        else:
            next_pos = 1
    elif index == val_list.size - 1:
        if val_list[index] < value:
            if not negative_activation:
                return membership[index]
            else:
                return -1 * (1 - membership[index])
        else:
            prev_pos = index - 1
            next_pos = index
    else:
        if (value - val_list[index]) > 0:
            prev_pos = index
            next_pos = index + 1
        else:
            prev_pos = index - 1
            next_pos = index

    sign = 1.0
    if value != 0:
        sign = value / abs(value)
    value = abs(value)

    # f(Xi) = (f(Xi-1)*Xi/Xi-1)*Xi-1_Xi_coef + (f(Xi+1)*Xi/Xi+1)*Xi+1_Xi_coef
    # inf_estimation = (membership[prev_pos] * value) / float(val_list[prev_pos])
    # sup_estimation = (membership[next_pos] * value) / float(val_list[next_pos])
    inf_estimation = membership[prev_pos]
    sup_estimation = membership[next_pos]
    diff = val_list[next_pos] - val_list[prev_pos]
    # calc influence coefficents
    inf_coef = 1 - ((value - val_list[prev_pos]) / diff)
    # 1 - inf_coef
    sup_coef = 1 - ((val_list[next_pos] - value) / diff)
    # result estimation according to distance between extremes
    estimation = sign * ((inf_coef * inf_estimation) + (sup_coef * sup_estimation))

    if not negative_activation:
        if estimation > 1:
            estimation = 1
        if estimation < -1:
            estimation = -1
        return estimation
    return -1 * (1 - estimation)


@njit
def exec_actv_function(function_id: int, val: float, args=np.empty(1, dtype=np.float64)) -> float:
    if function_id == FUNC_SATURATION:
        return saturation(val)
    if function_id == FUNC_BISTATE:
        return bistate(val)
    if function_id == FUNC_THREESTATE:
        return threestate(val)
    if function_id == FUNC_GCEQ:
        if args.size == 0:
            return greater_cond_equality(val)
        else:
            return greater_cond_equality(val, weight=args[0])
    if function_id == FUNC_LCEQ:
        if args.size == 0:
            return lower_cond_equality(val)
        else:
            return lower_cond_equality(val, weight=args[0])
    if function_id == FUNC_SIGMOID:
        if args.size == 0:
            return sigmoid(val)
        else:
            return sigmoid(val, lambda_val=args[0])
    if function_id == FUNC_SIGMOID_HIP:
        if args.size == 0:
            return sigmoid_hip(val)
        else:
            return sigmoid_hip(val, lambda_val=args[0])
    if function_id == FUNC_FUZZY:
        membership = args[:int(args.size / 2)]
        val_list = args[int(args.size / 2):]
        return fuzzy_set(val, membership, val_list)


# ensure  vectorized_run compilation
__empt_arr = np.ones(2, np.float64)
__empt_mat = np.ones((2, 2), np.float64)
vectorized_run(__empt_arr, __empt_mat, __empt_arr, List([__empt_arr, __empt_arr]),
               __empt_arr, List([True, False]), List([True, False]),
               max_iterations=3, min_diff=0.0001, extra_steps=0)

# ensure activation functions numba compilation
sigmoid(10, 1.5)
sigmoid_lambda(500, 0.8)
sigmoid_hip(10)
sigmoid_hip_lambda(500, 0.85)
bistate(10)
threestate(10)
saturation(10)
greater_cond_equality(10, 0.5)
fuzzy_set(10, np.array([0.0, 1.0]), np.array([5, 15]))
exec_actv_function(2, 10, np.array([2.0]))


class Activation:
    """
    Class to map all activation functions that can be used by FCM concepts. The function args structure will be the
     next one: val, arg_list
         Where:
           val: is the value to apply the function
           arg_list: is a numpy array that contains the list of arguments values sorted
    Note: If some function require more than one argument will be assumed that the values will be sorted according to
     the alphabetical sort of arguments names
    """

    @staticmethod
    def get_function_by_name(func_name: str):
        """
        Get the function callable object from the function name
        Args:
            func_name: Activation function name

        Returns: Function callable object if func_name is found, None otherwise

        """
        if func_name == "biestate":
            return bistate
        if func_name == "threestate":
            return threestate
        if func_name == "saturation":
            return saturation
        if func_name == "tan_hip":
            return sigmoid_hip
        if func_name == "sigmoid":
            return sigmoid
        if func_name == "sigmoid_hip":
            return sigmoid_hip
        if func_name == "fuzzy":
            return fuzzy_set
        if func_name == "gceq":
            return greater_cond_equality
        if func_name == "lceq":
            return lower_cond_equality
        return None

    @staticmethod
    def get_const_by_name(func_name: str):
        """
        Get the function const value from the function name
        Args:
            func_name: Activation function name

        Returns: Function cont value if func_name is found, None otherwise

        """
        if func_name == "biestate":
            return FUNC_BISTATE
        if func_name == "threestate":
            return FUNC_THREESTATE
        if func_name == "saturation":
            return FUNC_SATURATION
        if func_name == "tan_hip":
            return FUNC_SIGMOID_HIP
        if func_name == "sigmoid":
            return FUNC_SIGMOID
        if func_name == "sigmoid_hip":
            return FUNC_SIGMOID_HIP
        if func_name == "fuzzy":
            return FUNC_FUZZY
        if func_name == "gceq":
            return FUNC_GCEQ
        if func_name == "lceq":
            return FUNC_LCEQ
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
        names.add("gceq")
        names.add("lceq")
        names.add("proportion")
        return names


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
