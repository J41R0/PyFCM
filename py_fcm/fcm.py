import json
import random

import networkx as nx
from pandas import DataFrame
import matplotlib.pyplot as plt

from py_fcm.utils.functions import *


class FuzzyCognitiveMap:

    def __init__(self, max_it=200, extra_steps=5, stabilize=True, stability_diff=0.001, decision_function="MEAN",
                 mem_influence=True, activation_function="sigmoid_hip", **kwargs):
        """
        Fuzzy Cognitive Map Class, may be used like estimator.
        Args:
            max_it: Max map iterations
            extra_steps: Extra inference steps after end iterations
            stabilize: Exec map until stabilize
            stability_diff: Stability threshold value
            decision_function: Method for select winner decision node by its values during execution.
                LAST: Last inference node value
                MEAN: Highest average of all execution values in decision nodes
                EXITED: Highest last execution value in decision nodes
            activation_function: Activation function for map nodes
                biestate: Dual estate function => {0,1}
                threestate: Three  estate function => {0,0.5,1}
                saturation: Values lower than 0 or highest than 1 are transformed => [0;1]
                sigmoid: Sigmoid function => [0;1]
                sigmoid_hip: Hyperbolic sigmoid function => [-1;1]
                gceq: greater conditional equality => [-1,1]
                lceq: lower conditional equality=> [-1,1]
            mem_influence: Use or not memory influence in inference
            general_activ: Estimate values of arguments in activation functions according to node inputs values
            **kwargs: Activation function arguments, depend of function
        """
        # Map topology: {"<node_name>":{NODE_ACTIVE:<True/False>,NODE_ARCS:[("<node_name>",<ark_float_value>),...]
        #  ,NODE_AUX:<list>,NODE_VALUE:<v>,NODE_TYPE:<type of node>,NODE_USE_MEM:<True/False>,NODE_EXEC_FUNC:<callable>,
        # NODE_ACTV_FUNC:<callable>,NODE_ACTV_FUNC_ARGS:<kwargs>},NODE_TRAIN_ACTIVATION:<dict>}
        self.__topology = {}
        # Relation list copy for less inference complexity
        self.__arc_list = []
        # Execution behavior: {"<node_name>":[<values_list>], ...}
        self.__execution = {}
        # Stabilizing flag
        self.flag_stop_at_stabilize = stabilize
        # Maximum number of iterations
        self.max_iter = max_it
        # Extra steps after achieve stable state
        if extra_steps > 0:
            self.__extra_steps = extra_steps
        else:
            self.__extra_steps = 1
        # Stability threshold
        self.stability_diff = stability_diff
        # Activation when concept is found, may be certainty
        # self.init_activation = init_activ
        # TODO: parametrize function per output feature
        # Function to compute decision or regresor nodes, last
        self.__map_decision_function = None
        self.__decision_function = ''
        self.set_map_decision_function(decision_function)
        # memory influence flag
        self.flag_mem_influence = mem_influence
        # Activation function definition
        self.__map_activation_function = None
        self.__activation_function = ''

        # Set global function and arguments
        self.__global_func_args = None
        self.set_map_activation_function(activation_function, **kwargs)
        # map iterations
        self.__iterations = 1

        # str separator for datasets
        self.__separator = None

        # map weight for join process
        self.weight = 1
        self._default_concept = {}
        self.set_default_concept_properties()

        # if is false execution data will not be stored
        self.debug = True
        # fit inclination for sigmoid or sigmoid_hip functions
        self.fit_inclination = None

        self.__prepared_data = False
        self.__relation_matrix = None
        self.__state_vector = None
        self.__functions = []
        self.__memory_usage = []
        self.__function_args = []
        self.__avoid_saturation = []

    def set_map_decision_function(self, function_name: str):
        # Decision function definition
        new_function = Decision.get_by_name(function_name)
        if new_function is None:
            raise Exception("Unknown decision function '" + str(function_name) + "'")
        self.__decision_function = function_name
        self.__map_decision_function = new_function

    def set_map_activation_function(self, function_name: str, **kwargs):
        self.__prepared_data = False
        # Activation function definition
        new_function = Activation.get_function_by_name(function_name)
        if new_function is None:
            raise Exception("Unknown activation function '" + str(function_name) + "'")
        self.__activation_function = function_name
        self.__map_activation_function = new_function
        self.__global_func_args = kwargs
        for concept_name in self.__topology:
            if self.__topology[concept_name][NODE_USE_MAP_FUNC]:
                self.__topology[concept_name][NODE_ACTV_FUNC] = self.__map_activation_function
                self.__topology[concept_name][NODE_ACTV_FUNC_NAME] = self.__activation_function
                self.__topology[concept_name][NODE_ACTV_FUNC_ARGS] = self.__global_func_args
                self.__topology[concept_name][NODE_ACTV_FUNC_ARGS_VECT] = FuzzyCognitiveMap.__process_function_args(
                    self.__global_func_args)

    @staticmethod
    def __process_function_args(args_dict: dict):
        arg_names = [x for x in args_dict.keys()]
        arg_names.sort()
        values = []
        for arg in arg_names:
            if type(args_dict[arg]) == str:
                values.append(args_dict[arg])
            else:
                try:
                    iterator = iter(args_dict[arg])
                    for val in args_dict[arg]:
                        values.append(val)
                except TypeError:
                    values.append(args_dict[arg])
        return np.array(values, dtype=np.float64)

    @staticmethod
    def __process_activation_dict(activation_dict):
        # sort and scale activation dict
        val_list = np.array(activation_dict['val_list'], dtype=np.float64)
        membership = np.array(activation_dict['membership'], dtype=np.float64)
        scaled_val_list = np.zeros((len(val_list),), dtype=np.float64)
        dual_quick_sort(val_list, 0, len(val_list) - 1, membership)
        min_val = abs(val_list[0])
        max_val = val_list[len(val_list) - 1]
        for val_pos in range(len(val_list)):
            scaled_val_list[val_pos] = (val_list[val_pos] + min_val) / (min_val + max_val)
        new_activation_dict = {'val_list': val_list, 'membership': membership}
        scaled_activation_dict = {'val_list': scaled_val_list, 'membership': membership}
        return new_activation_dict, scaled_activation_dict

    def add_concept(self, concept_name: str, concept_type=None, is_active=None, use_memory=None,
                    excitation_function=None, activation_dict=None, activation_function=None, **kwargs):
        """
        Add new concept to map
        Args:
            concept_name: Concept name
            concept_type: Define type of node and behavior
            is_active: Define if node is active or not
            use_memory: Use memory in activation node process
            excitation_function: Custom function name for execution process
            activation_function: Function name for node activation, if none set default defined function
            activation_dict: activation dic for cont concepts according to found clusters in way =>
                {'membership': [], 'val_list': []} and related by position.
            **kwargs: arguments for activation function

        Returns: None

        Exitation functions:
        * "KOSKO": B. Kosko proposed activation function
        * "PAPAGEORGIUS": E. Papageorgius proposed function to avoid saturation

        Activation functions:
        * "saturation": 1 if value is > 1, 0 if values is < 0 and value otherwise. Domain => [0,1]
        * "biestate": 1 if value is > 0, 0 otherwise. Domain => {0,1}
        * "threestate": 0 if value is < 0.25, 0.5 if 0.25 <= value <= 0.75, 1 otherwise. Domain => {0,0.5,1}
        * "gceq": weight(float), return value if >= weight, 0 otherwise. Domain => [-1,1]
        * "lceq": weight(float), return value if <= weight, 0 otherwise. Domain => [-1,1]
        * "sigmoid": lambda_val(int), sigmoid function => [0,1]
        * "sigmoid_hip": lambda_val(int), sigmoid hyperbolic function => [-1,1]
        * "fuzzy" : fuzzy set activation, is set when is a fuzzy node type and an activation_dict are provided
        """
        self.__prepared_data = False
        self.__topology[concept_name] = {NODE_ARCS: [], NODE_AUX: [], NODE_VALUE: 0.0, NODE_ACTV_SUM: 0.0}

        # TODO: reduce duplicated code
        if is_active is not None and type(is_active) == bool:
            self.__topology[concept_name][NODE_ACTIVE] = is_active
        else:
            self.__topology[concept_name][NODE_ACTIVE] = self._default_concept[NODE_ACTIVE]

        if excitation_function is not None:
            function = Excitation.get_by_name(excitation_function)
            if function is not None:
                self.__topology[concept_name][NODE_EXEC_FUNC] = function
                self.__topology[concept_name][NODE_EXEC_FUNC_NAME] = excitation_function
            else:
                raise Exception("Unknown excitation function: " + excitation_function)
        else:
            self.__topology[concept_name][NODE_EXEC_FUNC] = self._default_concept[NODE_EXEC_FUNC]
            self.__topology[concept_name][NODE_EXEC_FUNC_NAME] = self._default_concept[NODE_EXEC_FUNC_NAME]

        if is_valid_type(concept_type) and concept_type is not None:
            self.__topology[concept_name][NODE_TYPE] = concept_type
        else:
            self.__topology[concept_name][NODE_TYPE] = self._default_concept[NODE_TYPE]

        # scale and normalize the values for fuzzy function
        # activation_dict = {'membership':[],'val_list':[]}
        if (self.__topology[concept_name][NODE_TYPE] == TYPE_FUZZY or
                self.__topology[concept_name][NODE_TYPE] == TYPE_REGRESOR):
            if activation_dict is not None:
                new_activation_dict, scaled_activation_dict = FuzzyCognitiveMap.__process_activation_dict(
                    activation_dict)
                self.__topology[concept_name][NODE_FUZZY_ACTIVATION] = new_activation_dict
                self.__topology[concept_name][NODE_FUZZY_MIN] = abs(new_activation_dict['val_list'][0])
                activation_elements = len(new_activation_dict['val_list'])
                self.__topology[concept_name][NODE_FUZZY_MAX] = new_activation_dict['val_list'][activation_elements - 1]
                activation_dict = scaled_activation_dict
            elif self._default_concept[NODE_TYPE] == self.__topology[concept_name][NODE_TYPE]:
                self.__topology[concept_name][NODE_FUZZY_ACTIVATION] = self._default_concept[NODE_FUZZY_ACTIVATION]
                self.__topology[concept_name][NODE_FUZZY_MIN] = self._default_concept[NODE_FUZZY_MIN]
                self.__topology[concept_name][NODE_FUZZY_MAX] = self._default_concept[NODE_FUZZY_MAX]

        if use_memory is not None and type(use_memory) == bool:
            self.__topology[concept_name][NODE_USE_MEM] = use_memory
        else:
            self.__topology[concept_name][NODE_USE_MEM] = self._default_concept[NODE_USE_MEM]

        # define activation function
        self.__topology[concept_name][NODE_USE_MAP_FUNC] = False
        if activation_dict is not None:
            self.__topology[concept_name][NODE_ACTV_FUNC] = Activation.get_function_by_name("fuzzy")
            self.__topology[concept_name][NODE_ACTV_FUNC_NAME] = 'fuzzy'
            self.__topology[concept_name][NODE_ACTV_FUNC_ARGS] = activation_dict
            vec_actv_funct_args = FuzzyCognitiveMap.__process_function_args(activation_dict)
        elif activation_function is not None:
            actv_function = Activation.get_function_by_name(activation_function)
            if actv_function is not None:
                self.__topology[concept_name][NODE_ACTV_FUNC] = actv_function
                self.__topology[concept_name][NODE_ACTV_FUNC_NAME] = activation_function
                self.__topology[concept_name][NODE_ACTV_FUNC_ARGS] = kwargs
                vec_actv_funct_args = FuzzyCognitiveMap.__process_function_args(kwargs)
            else:
                raise Exception("Unknown activation function: " + activation_function)
        else:
            self.__topology[concept_name][NODE_ACTV_FUNC] = self._default_concept[NODE_ACTV_FUNC]
            self.__topology[concept_name][NODE_ACTV_FUNC_NAME] = self._default_concept[NODE_ACTV_FUNC_NAME]
            self.__topology[concept_name][NODE_ACTV_FUNC_ARGS] = self._default_concept[NODE_ACTV_FUNC_ARGS]
            vec_actv_funct_args = FuzzyCognitiveMap.__process_function_args(self._default_concept[NODE_ACTV_FUNC_ARGS])
            self.__topology[concept_name][NODE_USE_MAP_FUNC] = True
        self.__topology[concept_name][NODE_ACTV_FUNC_ARGS_VECT] = vec_actv_funct_args
        self.__execution[concept_name] = [0.0]
        self.__topology[concept_name][NODE_VALUE] = 0.0

    def set_concept_properties(self, concept_name: str, concept_type=None, is_active=None, use_memory=None,
                               excitation_function=None, activation_dict=None, activation_function=None, **kwargs):
        if concept_name in self.__topology:
            self.__prepared_data = False
            if is_active is not None and type(is_active) == bool:
                self.__topology[concept_name][NODE_ACTIVE] = is_active
            if excitation_function is not None:
                new_function = Excitation.get_by_name(excitation_function)
                if new_function is not None:
                    self.__topology[concept_name][NODE_EXEC_FUNC] = new_function
                    self.__topology[concept_name][NODE_EXEC_FUNC_NAME] = excitation_function
                else:
                    raise Exception("Unknown excitation function: " + excitation_function)
            if concept_type is not None:
                if is_valid_type(concept_type):
                    self.__topology[concept_name][NODE_TYPE] = concept_type
                else:
                    raise Exception("Unknown node type")

            # scale and normalize the values for fuzzy function
            # activation_dict = {'membership':[],'val_list':[]}
            if (concept_type == TYPE_FUZZY or concept_type == TYPE_REGRESOR) and activation_dict is not None:
                new_activation_dict, scaled_activation_dict = FuzzyCognitiveMap.__process_activation_dict(
                    activation_dict)
                self.__topology[concept_name][NODE_FUZZY_ACTIVATION] = new_activation_dict
                self.__topology[concept_name][NODE_FUZZY_MIN] = abs(new_activation_dict['val_list'][0])
                activation_elements = len(new_activation_dict['val_list'])
                self.__topology[concept_name][NODE_FUZZY_MAX] = new_activation_dict['val_list'][activation_elements - 1]
                activation_dict = scaled_activation_dict

            if use_memory is not None and type(use_memory) == bool:
                self.__topology[concept_name][NODE_USE_MEM] = use_memory
            else:
                self.__topology[concept_name][NODE_USE_MEM] = self.flag_mem_influence

            # define activation function
            self.__topology[concept_name][NODE_USE_MAP_FUNC] = False
            if activation_dict is not None:
                self.__topology[concept_name][NODE_ACTV_FUNC] = Activation.get_function_by_name("fuzzy")
                self.__topology[concept_name][NODE_ACTV_FUNC_NAME] = 'fuzzy'
                self.__topology[concept_name][NODE_ACTV_FUNC_ARGS] = activation_dict
                vec_actv_funct_args = FuzzyCognitiveMap.__process_function_args(activation_dict)
            elif activation_function is not None:
                actv_function = Activation.get_function_by_name(activation_function)
                if actv_function is not None:
                    self.__topology[concept_name][NODE_ACTV_FUNC] = actv_function
                    self.__topology[concept_name][NODE_ACTV_FUNC_NAME] = activation_function
                    self.__topology[concept_name][NODE_ACTV_FUNC_ARGS] = kwargs
                    vec_actv_funct_args = FuzzyCognitiveMap.__process_function_args(kwargs)
                else:
                    raise Exception("Unknown activation function: " + activation_function)
            else:
                self.__topology[concept_name][NODE_ACTV_FUNC] = self.__map_activation_function
                self.__topology[concept_name][NODE_ACTV_FUNC_NAME] = self.__activation_function
                self.__topology[concept_name][NODE_ACTV_FUNC_ARGS] = self.__global_func_args
                vec_actv_funct_args = FuzzyCognitiveMap.__process_function_args(self.__global_func_args)
                self.__topology[concept_name][NODE_USE_MAP_FUNC] = True
            self.__topology[concept_name][NODE_ACTV_FUNC_ARGS_VECT] = vec_actv_funct_args
        else:
            raise Exception("Concept " + concept_name + " not found")

    def set_default_concept_properties(self, concept_type=TYPE_SIMPLE, is_active=True, use_memory=None,
                                       excitation_function='KOSKO', activation_dict=None, activation_function=None,
                                       **kwargs):
        if is_active is not None and type(is_active) == bool:
            self._default_concept[NODE_ACTIVE] = is_active

        new_function = Excitation.get_by_name(excitation_function)
        if new_function is not None:
            self._default_concept[NODE_EXEC_FUNC] = new_function
            self._default_concept[NODE_EXEC_FUNC_NAME] = excitation_function
        else:
            raise Exception("Unknown excitation function: " + excitation_function)
        if is_valid_type(concept_type):
            self._default_concept[NODE_TYPE] = concept_type
        else:
            raise Exception("Unknown node type")

        if concept_type == TYPE_FUZZY or concept_type == TYPE_REGRESOR and activation_dict is not None:
            new_activation_dict, scaled_activation_dict = FuzzyCognitiveMap.__process_activation_dict(activation_dict)
            self._default_concept[NODE_FUZZY_ACTIVATION] = new_activation_dict
            self._default_concept[NODE_FUZZY_MIN] = abs(new_activation_dict['val_list'][0])
            activation_elements = len(new_activation_dict['val_list'])
            self._default_concept[NODE_FUZZY_MAX] = new_activation_dict['val_list'][activation_elements - 1]
            activation_dict = scaled_activation_dict

        if use_memory is not None and type(use_memory) == bool:
            self._default_concept[NODE_USE_MEM] = use_memory
        else:
            self._default_concept[NODE_USE_MEM] = self.flag_mem_influence

        # define activation function
        self._default_concept[NODE_USE_MAP_FUNC] = False
        if activation_dict is not None:
            self._default_concept[NODE_ACTV_FUNC] = Activation.get_function_by_name("fuzzy")
            self._default_concept[NODE_ACTV_FUNC_NAME] = 'fuzzy'
            self._default_concept[NODE_ACTV_FUNC_ARGS] = activation_dict
        elif activation_function is not None:
            actv_function = Activation.get_function_by_name(activation_function)
            if actv_function is not None:
                self._default_concept[NODE_ACTV_FUNC] = actv_function
                self._default_concept[NODE_ACTV_FUNC_NAME] = activation_function
                self._default_concept[NODE_ACTV_FUNC_ARGS] = kwargs
            else:
                raise Exception("Unknown ativation function: " + activation_function)
        else:
            self._default_concept[NODE_ACTV_FUNC] = self.__map_activation_function
            self._default_concept[NODE_ACTV_FUNC_NAME] = self.__activation_function
            self._default_concept[NODE_ACTV_FUNC_ARGS] = self.__global_func_args
            self._default_concept[NODE_USE_MAP_FUNC] = True

    def get_concept_value(self, concept_name: str):
        if concept_name in self.__topology:
            return self.__topology[concept_name][NODE_VALUE]

    def add_relation(self, origin_concept: str, destiny_concept: str, weight: float):
        """
        Add relation between existent pair of nodes
        Args:
            origin_concept: Origin node name
            destiny_concept: Destiny node name
            weight: Relation weight

        Returns: None

        """
        self.__prepared_data = False
        if destiny_concept in self.__topology and origin_concept in self.__topology:
            # TODO: NODE_ARCS usages deprecation
            self.__topology[origin_concept][NODE_ARCS].append((destiny_concept, weight))
            self.__topology[destiny_concept][NODE_ACTV_SUM] += weight

            # TODO: refactor fit inclination behavior
            if self.fit_inclination is not None:
                new_lambda = None
                if self.__topology[destiny_concept][NODE_ACTV_FUNC_NAME] == "sigmoid":
                    new_lambda = sigmoid_lambda(abs(self.__topology[destiny_concept][NODE_ACTV_SUM]),
                                                self.fit_inclination)

                if self.__topology[destiny_concept][NODE_ACTV_FUNC_NAME] == "sigmoid_hip":
                    new_lambda = sigmoid_hip_lambda(abs(self.__topology[destiny_concept][NODE_ACTV_SUM]),
                                                    self.fit_inclination)

                if new_lambda is not None:
                    new_args = {"lambda_val": new_lambda}
                    self.__topology[destiny_concept][NODE_ACTV_FUNC_ARGS] = new_args
                    vec_actv_func_args = FuzzyCognitiveMap.__process_function_args(new_args)
                    self.__topology[destiny_concept][NODE_ACTV_FUNC_ARGS_VECT] = vec_actv_func_args

            self.__arc_list.append((destiny_concept, weight, origin_concept))

    def get_concept_outgoing_relations(self, concept_name):
        if concept_name in self.__topology:
            return self.__topology[concept_name][NODE_ARCS].copy()
        return []

    def clear_all(self):
        """
        Reset whole data in map
        Returns: None

        """
        self.reset_execution()
        self.__topology = {}
        self.__arc_list = []
        self.__execution = {}

    def init_concept(self, concept_name: str, value, required_presence=True):
        """

        Set concept initial value, by default 1
        Args:
            concept_name: Concept name
            value: Feature value
            required_presence: Is required the concept presence
        Returns: None

        """
        if concept_name not in self.__topology:
            if required_presence:
                raise Exception("Missing concept " + concept_name)
            else:
                return
        if self.__topology[concept_name][NODE_ACTV_FUNC_NAME] == 'fuzzy':
            membership = self.__topology[concept_name][NODE_FUZZY_ACTIVATION]['membership']
            val_list = self.__topology[concept_name][NODE_FUZZY_ACTIVATION]['val_list']
            if type(membership) == list or type(val_list) == list:
                membership = np.array(membership)
                val_list = np.array(val_list)
            fuzzy_actv_function = Activation.get_function_by_name('fuzzy')
            actv_value = fuzzy_actv_function(value, membership=membership, val_list=val_list)
            self.__topology[concept_name][NODE_VALUE] = actv_value
            self.__execution[concept_name] = [actv_value]
        else:
            self.__topology[concept_name][NODE_VALUE] = value
            self.__execution[concept_name] = [value]
        self.__topology[concept_name][NODE_AUX] = []

    def is_stable(self):
        """
        Determine if whole inference system is stable
        Returns: True if system is stable, False otherwise

        """
        # all(all nodes) or result(decision or target nodes)
        stable = True
        if self.__iterations > 1:
            for node in self.__execution.keys():
                # difference between last two states
                if abs(self.__execution[node][-1] - self.__execution[node][-2]) > self.stability_diff:
                    stable = False
        else:
            return False
        return stable

    def reset_execution(self):
        """
        Restart map execution to rerun inference process
        Returns: None

        """
        for concept, data in self.__topology.items():
            data[NODE_AUX] = []
            data[NODE_VALUE] = self.__execution[concept][0]
            self.__execution[concept] = [data[NODE_VALUE]]

    def clear_execution(self):
        """
        Clear map execution and inital concepts values for a new inference process
        Returns: None

        """
        for concept, data in self.__topology.items():
            data[NODE_AUX] = []
            data[NODE_VALUE] = 0.0
            self.__execution[concept] = [0.0]

    def __iterative_inference(self):
        self.__iterations = 1
        extra_steps = self.__extra_steps - 1
        while extra_steps > 0:
            # execute extra_steps new iterations after finish execution
            if not self.__keep_execution():
                extra_steps -= 1
            else:
                extra_steps = self.__extra_steps - 1
            for arc in self.__arc_list:
                origin = arc[RELATION_ORIGIN]
                dest = arc[RELATION_DESTINY]
                weight = arc[RELATION_WEIGHT]
                # set value to: sum(wij * Ai)
                if self.__topology[origin][NODE_ACTIVE]:
                    if self.__topology[origin][NODE_EXEC_FUNC_NAME] == "PAPAGEORGIUS":
                        self.__topology[dest][NODE_AUX].append((2 * self.__topology[origin][NODE_VALUE] - 1) * weight)
                    else:
                        self.__topology[dest][NODE_AUX].append(self.__topology[origin][NODE_VALUE] * weight)

            for node in self.__execution.keys():
                # calc execution value using NODE_AUX, NODE_VALUE, NODE_USE_MEM and NODE_ACTIVE, the values may change
                exec_val = self.__topology[node][NODE_EXEC_FUNC](self.__topology[node])
                # reset NODE_AUX after calculate the execution value
                self.__topology[node][NODE_AUX].clear()  # = []
                # normalize values for fuzzy activation nodes
                if self.__topology[node][NODE_ACTV_FUNC_NAME] == 'fuzzy':
                    exec_val = exec_val / self.__topology[node][NODE_ACTV_SUM]
                result = self.__topology[node][NODE_ACTV_FUNC](exec_val, **self.__topology[node][NODE_ACTV_FUNC_ARGS])
                # update execution values
                self.__topology[node][NODE_VALUE] = result
                self.__execution[node].append(result)

            self.__iterations += 1

    def __vectorized_inference(self):
        concepts = list(self.__topology.keys())
        if not self.__prepared_data:
            self.__prepared_data = True
            self.__relation_matrix = np.zeros((len(self.__topology), len(self.__topology)), dtype=np.float64)
            self.__state_vector = np.zeros(len(self.__topology), dtype=np.float64)
            self.__functions = np.zeros(len(self.__topology), dtype=np.int32)
            self.__memory_usage = []
            self.__normalize_values = np.zeros(len(self.__topology), dtype=np.float64)
            self.__function_args = []
            self.__avoid_saturation = []

            for concept_pos in range(len(concepts)):
                self.__state_vector[concept_pos] = self.__topology[concepts[concept_pos]][NODE_VALUE]
                self.__functions[concept_pos] = Activation.get_const_by_name(
                    self.__topology[concepts[concept_pos]][NODE_ACTV_FUNC_NAME])
                self.__function_args.append(self.__topology[concepts[concept_pos]][NODE_ACTV_FUNC_ARGS_VECT])
                self.__memory_usage.append(self.__topology[concepts[concept_pos]][NODE_USE_MEM])
                # normalize fuzzy activation input
                if self.__topology[concepts[concept_pos]][NODE_ACTV_FUNC_NAME] == 'fuzzy':
                    self.__normalize_values[concept_pos] = self.__topology[concepts[concept_pos]][NODE_ACTV_SUM]

                if 'PAPAGEORGIUS' == self.__topology[concepts[concept_pos]][NODE_EXEC_FUNC_NAME]:
                    self.__avoid_saturation.append(True)
                else:
                    self.__avoid_saturation.append(False)
            for arc in self.__arc_list:
                origin = arc[RELATION_ORIGIN]
                dest = arc[RELATION_DESTINY]
                weight = arc[RELATION_WEIGHT]
                origin_index = concepts.index(origin)
                dest_index = concepts.index(dest)
                self.__relation_matrix[origin_index][dest_index] = weight
        # print(self.__relation_matrix)
        # run vectorized jit inference process
        result = vectorized_run(self.__state_vector, self.__relation_matrix, self.__functions,
                                List(self.__function_args), self.__normalize_values, List(self.__memory_usage),
                                List(self.__avoid_saturation), np.int32(self.max_iter),
                                np.float64(self.stability_diff), np.int32(self.__extra_steps))
        self.__state_vector = result[:, 0]
        col_index = self.max_iter - 1
        for val_pos in range(self.max_iter):
            if result[0, val_pos] == 2.0:
                col_index = val_pos
                break

        for concept_pos in range(len(concepts)):
            self.__execution[concepts[concept_pos]] = result[concept_pos, :col_index].tolist()

    def run_inference(self, reset=True):
        """
        Execute map inference process
        Returns: None

        """
        if reset:
            self.reset_execution()
        # print(self.to_json())
        if self.debug:
            self.__iterative_inference()
        else:
            self.__vectorized_inference()

        for node in self.__topology:
            self.__topology[node][NODE_VALUE] = self.__map_decision_function(self.__execution[node])

    def get_final_state(self, concept_type="target", names=[]):
        """
        Get inference result values of node_type or all nodes
        Args:
            concept_type: Type of nodes inference result
                "any": calc all nodes final state
                "target": calc only DECISION or REGRESSION node types final state
            names: concept name list

        Returns: Dict in way: {"<node_id>": <final_value>}

        """
        result = {}
        if len(names) > 0:
            for concept, exec_values in self.__execution.items():
                if concept in names:
                    result[concept] = self.__map_decision_function(exec_values)
        else:
            for concept_id in self.__execution.keys():
                if concepts_type == "any":
                    result[concept_id] = self.__map_decision_function(self.__execution[concept_id])
                if concepts_type == "target":
                    if (self.__topology[concept_id][NODE_TYPE] == TYPE_DECISION
                            or self.__topology[concept_id][NODE_TYPE] == TYPE_REGRESOR):
                        result[concept_id] = self.__map_decision_function(self.__execution[concept_id])
                elif self.__topology[concept_id][NODE_TYPE] == concepts_type:
                    result[concept_id] = self.__map_decision_function(self.__execution[concept_id])
        return result

    def plot_execution(self, fig_name="map", limit=0, plot_all=False, path=""):
        colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']
        for node in self.__execution.keys():
            end = len(self.__execution[node])
            if end > limit > 0:
                end = limit
            plot_val = []
            for it in range(0, end):
                plot_val.append(self.__execution[node][it])
            if self.__topology[node][NODE_TYPE] == TYPE_DECISION or self.__topology[node][
                NODE_TYPE] != TYPE_FUZZY or plot_all:
                plt.plot(range(0, end), plot_val, colors[random.randint(a=0, b=9)])
                # plt.axis([0, len(current.values) - 1, 0.0, 1.0])
                plt.axis([0, end, -1.1, 1.1])

        plt.savefig(path + fig_name + '.png')
        plt.close()

    def plot_topology(self, path='', fig_name="topology"):
        graph = nx.Graph()
        graph.add_nodes_from(self.__topology.keys())
        for key in self.__topology.keys():
            for out_node in self.__topology[key][NODE_ARCS]:
                graph.add_edge(key, out_node[RELATION_DESTINY])
        nx.draw(graph)
        plt.savefig(path + fig_name + ".png")
        plt.close()

    def to_json(self):
        """
        Generate the output JSON for current map

        Returns: String in JSON format

        """
        result = {
            'max_iter': self.max_iter,
            'decision_function': self.__decision_function,
            'activation_function': self.__activation_function,
            'memory_influence': self.flag_mem_influence,
            'stability_diff': self.stability_diff,
            'stop_at_stabilize': self.flag_stop_at_stabilize,
            'extra_steps': self.__extra_steps,
            'weight': self.weight,
            'concepts': [],
            'relations': []
        }
        if len(self.__global_func_args) > 0:
            result['activation_function_args'] = self.__global_func_args

        for concept_name in self.__topology:
            # TODO: add all supported types
            type_name = 'SIMPLE'
            if self.__topology[concept_name][NODE_TYPE] == TYPE_DECISION:
                type_name = 'DECISION'
            if self.__topology[concept_name][NODE_TYPE] == TYPE_FUZZY:
                type_name = 'FUZZY'
            if self.__topology[concept_name][NODE_TYPE] == TYPE_REGRESOR:
                type_name = 'REGRESSION'
            concept_desc = {
                'id': concept_name,
                'is_active': self.__topology[concept_name][NODE_ACTIVE],
                'type': type_name,
                'activation': self.__topology[concept_name][NODE_VALUE]
            }
            if self.__topology[concept_name][NODE_USE_MEM] != self.flag_mem_influence:
                concept_desc['use_memory'] = self.__topology[concept_name][NODE_USE_MEM]
            if self.__topology[concept_name][NODE_ACTV_FUNC_NAME] != self.__activation_function:
                concept_desc['custom_function'] = self.__topology[concept_name][NODE_ACTV_FUNC_NAME]
                if concept_desc['custom_function'] == 'fuzzy':
                    if type(self.__topology[concept_name][NODE_FUZZY_ACTIVATION]['membership']) == list:
                        concept_desc['activation_dict'] = {
                            'membership': self.__topology[concept_name][NODE_FUZZY_ACTIVATION]['membership'],
                            'val_list': self.__topology[concept_name][NODE_FUZZY_ACTIVATION]['val_list']
                        }
                    else:
                        concept_desc['activation_dict'] = {
                            'membership': self.__topology[concept_name][NODE_FUZZY_ACTIVATION]['membership'].tolist(),
                            'val_list': self.__topology[concept_name][NODE_FUZZY_ACTIVATION]['val_list'].tolist()
                        }
                else:
                    if len(self.__topology[concept_name][NODE_ACTV_FUNC_ARGS]) > 0:
                        concept_desc['custom_function_args'] = self.__topology[concept_name][NODE_ACTV_FUNC_ARGS]
            result['concepts'].append(concept_desc)

        for relation in self.__arc_list:
            relation_desc = {
                'origin': relation[RELATION_ORIGIN],
                'destiny': relation[RELATION_DESTINY],
                'weight': relation[RELATION_WEIGHT]
            }
            result['relations'].append(relation_desc)

        return json.dumps(result)

    # private functions
    def __fit(self, x, y):
        # TODO
        raise NotImplementedError("Require a FCM topology generator")

    def __score(self, x, y, plot=False):
        # TODO
        raise NotImplementedError("Scorer not defined")

    def __estimate(self):
        # group execution results by feature
        # exec_res = {'feature_name':[{'concept_name',[<exec_val_list>]}, ...]}
        exec_res = {}
        for concept in self.__execution.keys():
            if (self.__topology[concept][NODE_TYPE] == TYPE_DECISION or
                    self.__topology[concept][NODE_TYPE] == TYPE_REGRESOR):
                feat_name = str(concept).split(self.__separator)[-1]
                if feat_name in exec_res.keys():
                    exec_res[feat_name].append({concept: self.__execution[concept]})
                else:
                    exec_res[feat_name] = []
                    exec_res[feat_name].append({concept: self.__execution[concept]})

        # estimate result for each concept set associated to target feature
        result = {}
        for feat in exec_res.keys():
            value = None
            # TODO: estimate value for each target feature
            # all concepts in feature share the same type
            some_concept_name = list(exec_res[feat][0].keys())[0]

            # Estimate discrete result
            if self.__topology[some_concept_name][NODE_TYPE] == TYPE_DECISION:
                # solving by concept
                class_name = ""
                for concept_data in exec_res[feat]:
                    concept_name = list(concept_data.keys())[0]
                    if self.is_stable():
                        res = self.__last(self.__execution[concept_name])
                    else:
                        res = self.estimate_desc_func(self.__execution[concept_name])
                    # select gratest result value
                    if value is None:
                        value = res
                        class_name = str(concept_name).split(self.sep)[0]
                    elif value < res:
                        value = res
                        class_name = str(concept_name).split(self.sep)[0]
                        # the class is the result
                value = class_name

            # Estimate continous result
            if self.__topology[some_concept_name][NODE_TYPE] == TYPE_REGRESOR:
                candidates = []
                for concept_data in exec_res[feat]:
                    concept_name = list(concept_data.keys())[0]
                    res = self.estimate_desc_func(self.__execution[concept_name])
                    candidates.extend(Fuzzy.defuzzyfication(res,
                                                            self.__topology[concept_name][NODE_FUZZY_MIN],
                                                            self.__topology[concept_name][NODE_FUZZY_MAX],
                                                            **self.__topology[concept_name][NODE_FUZZY_ACTIVATION]))
                # search minimum size interval
                candidates.sort()
                if len(candidates) == 0:
                    raise Exception("Cannot estimate membership value " + str(res) + " in concept " + concept_name)
                # print(candidates)
                res = (0, len(candidates) - 1)
                size = candidates[-1] - candidates[0]
                for pos in range(len(candidates) - len(exec_res[feat])):
                    first = candidates[pos]
                    last = candidates[pos + len(exec_res[feat]) - 1]
                    if last - first <= size:
                        size = last - first
                        res = (pos, pos + len(exec_res[feat]) - 1)

                # define final result as mean of nearest estimation
                value = 0
                for pos in range(res[0], res[1] + 1):
                    value += candidates[pos]
                value = value / len(exec_res[feat])
            result[feat] = value
        return result

    def __predict(self, x, plot=False):
        result = []
        x_df = DataFrame(x)
        plot_it = 0
        for index, row in x_df.iterrows():
            plot_it += 1
            # init map from data
            self.clear_execution()
            for feature in x_df:
                self.reset_execution()
                try:
                    self.init_from_ds(feature, row[feature])
                except Exception as err:
                    raise Exception("Can not init concept related to feature '" + feature + "' due: " + str(err))
            self.run_inference()
            res = self.__estimate()
            res = res[list(res.keys())[0]]
            result.append(res)
            if plot:
                self.plot_execution(fig_name="map" + str(plot_it), limit=50)
        return result

    def __keep_execution(self):
        """
        Define if the inference process must go on
        Returns: True if needs other inference step, False otherwise

        """
        if self.__iterations >= self.max_iter:
            return False
        if self.flag_stop_at_stabilize:
            return not self.is_stable()
        return True

    def __find_related_concept_type(self, name):
        """
        Get the data related to first concept where <name> is a substring
        Args:
            name: concept name

        Returns: Found concept type, None otherwise

        """
        concepts_list = list(self.__topology.keys())
        for concept in concepts_list:
            if name in concept:
                return self.__topology[concept][NODE_TYPE]
        return None

    def __get_related_concepts_names(self, name):
        """
        Search for all concepts where <name> is a substring
        Args:
            name: concept name

        Returns: Associated concept list

        """
        result = []
        concepts_list = list(self.__topology.keys())
        for concept in concepts_list:
            if name in concept:
                result.append(concept)
        return result

    def __est_concept_func_args(self, concept, x_val, y_val):
        con_func = self.__topology[concept][NODE_ACTV_FUNC]
        kwargs = {}
        if con_func == Activation.sigmoid:
            lambda_val = Activation.sigmoid_lambda(x_val, y_val)
            kwargs["lambda_val"] = lambda_val

        if con_func == Activation.sigmoid_hip:
            lambda_val = Activation.sigmoid_hip_lambda(x_val, y_val)
            kwargs["lambda_val"] = lambda_val
        self.__topology[concept][NODE_ACTV_FUNC_ARGS] = kwargs

    def __init_from_ds(self, feature, value, separator='_'):
        """
        Init dataset feature associated concept
        Args:
            feature: feature name
            value: Feature value

        Returns: None

        """
        self.__separator = separator
        concept_type = self.__find_related_concept_type(feature)
        if concept_type:
            processed = False
            if concept_type != TYPE_REGRESOR and concept_type != TYPE_FUZZY and type(value) == str:
                # init all discrete concepts value, estimating the concept name
                disc_concept_name = str(value) + self.__separator + feature
                if self.__find_related_concept_type(disc_concept_name):
                    self.__topology[disc_concept_name][NODE_VALUE] = 1
                    self.__execution[disc_concept_name] = [1]
                    processed = True
                else:
                    Warning("Concept '" + disc_concept_name + "' not defined in FCM topology.")
            else:
                related_concepts = self.__get_related_concepts_names(feature)
                for concept in related_concepts:
                    activation_value = Fuzzy.fuzzyfication(value,
                                                           self.__topology[concept][NODE_FUZZY_MIN],
                                                           self.__topology[concept][NODE_FUZZY_MAX],
                                                           **self.__topology[concept][NODE_FUZZY_ACTIVATION])
                    self.__topology[concept][NODE_VALUE] = activation_value
                    self.__execution[concept] = [activation_value]
                processed = True
            if not processed:
                raise Exception(
                    "Error processing value " + str(value) + " from feature '" + feature + "' and type '" + str(
                        type(value)) + "'")
        else:
            raise Warning("Concepts related to feature '" + feature + "' are not defined in FCM topology.")
