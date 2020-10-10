import json
import random
from collections import defaultdict

import numpy as np
import networkx as nx
from pandas import DataFrame
import matplotlib.pyplot as plt

from py_fcm.functions import Activation, Excitation, Decision, Fuzzy
from py_fcm.__const import *


def from_json(str_json: str):
    """
    Function to genrate a FCM object form a JSON like:
    {
     "max_iter": 500,
     "activation_function": "sigmoid",
     "actv_func_args": {"lambda_val":1},
     "memory_influence": false,
     "decision_function": "LAST",
     "concepts" :
      [
        {"id": "concept_1", "type": "SIMPLE", "activation": 0.5},
        {"id": "concept_2", "type": "DECISION", "custom_function": "sum_w", "custom_function_args": {"weight":0.3}},
        {"id": "concept_3", "type": "SIMPLE", "memory_influence":true },
        {"id": "concept_4", "type": "SIMPLE", "custom_function": "saturation", "activation": 0.3}
      ],
     "relations":
      [
        {"origin": "concept_4", "destiny": "concept_2", "weight": -0.1},
        {"origin": "concept_1", "destiny": "concept_3", "weight": 0.59},
        {"origin": "concept_3", "destiny": "concept_2", "weight": 0.8911}
      ]
    }
    Structure:
    * iter: max map iterations, required.
    * activation_function: defalt activation function, required
    * activation_function_args: object (JSON serializable) to describe required function params, optional value
    * memory_influence: use memory or not, required
    * stability_diff: difference to consider a stable FCM state, optional with 0.001 by default
    * stop_at_stabilize: stop the inference process when the FCM reach a stable state, optional True by default
    * extra_steps: additional steps to execute after reach a stable state, optionay with 5 by default
    * weight: FCM weight ti be used in joint map process, optional with 1 by default
    * decision_function: define the decision function to get the final value, required:
        - "LAST": last inference value
        - "MEAN": whole execution average value
        - "EXITED": Highest last execution value in decision nodes
    * concepts: a concept list describing each concept, required
    * relations: a relations list between defined concepts, required

    Concept descrption:
    * id: concept id
    * type: node type => "SIMPLE": regular node and default ,"DECISION": target for a classification problems
    * active: define if node is active or not, by default is considered active
    * custom_function: custom node function, by default use map defined function
    * custom_function_args: object (JSON serializable) to describe custom_function params
    * memory_influence: use memory or not, by default use FCM memory definition
    * exitation_function: node exitation function, KOSKO by default
    * activation: initial node activation value, by default 0

    Relation descrption:
    * origin: start concept id
    * destiny: destiny concept id
    * weight: relaion weight in range => [-1,1]

    Exitation functions:
    * "MEAN": Mean values of all neighbors that influence the node
    * "KOSKO": B. Kosko proposed activation function
    * "PAPAGEORGIUS": E. Papageorgius proposed function to avoid saturation

    Activation functions:
    * "saturation": 1 if value is > 1, 0 if values is < 0 and value otherwise. Domain => [0,1]
    * "biestate": 1 if value is > 0, 0 otherwise. Domain => {0,1}
    * "threestate": 0 if value is < 0.25, 0.5 if 0.25 <= value <= 0.75, 1 otherwise. Domain => {0,0.5,1}
    * "sum_w": weight(float), return value if > weight, 0 otherwise. Domain => [-1,1]
    * "sigmoid": lambda_val(int), sigmoid function => [0,1]
    * "sigmoid_hip": lambda_val(int), sigmoid hyperbolic function => [-1,1]

    Args:
        str_json: string JSON

    Returns: FCM object

    """
    try:
        data_dict = json.loads(str_json)
        actv_param = {}
        stability_diff = 0.001
        stop_at_stabilize = True
        extra_steps = 5
        weight = 1
        if 'activation_function_args' in data_dict:
            actv_param = data_dict['activation_function_args']

        if 'stability_diff' in data_dict:
            stability_diff = data_dict['stability_diff']

        if 'stop_at_stabilize' in data_dict:
            stop_at_stabilize = data_dict['stop_at_stabilize']

        if 'extra_steps' in data_dict:
            extra_steps = data_dict['extra_steps']

        if 'weight' in data_dict:
            weight = data_dict['weight']

        my_fcm = FuzzyCognitiveMap(max_it=data_dict['max_iter'],
                                   decision_function=data_dict['decision_function'],
                                   mem_influence=data_dict['memory_influence'],
                                   activ_function=data_dict['activation_function'],
                                   stability_diff=stability_diff,
                                   stabilize=stop_at_stabilize,
                                   extra_steps=extra_steps,
                                   **actv_param)
        my_fcm.weight = weight
        # adding concepts
        for concept in data_dict['concepts']:
            use_mem = None
            if 'memory_influence' in concept:
                use_mem = concept['memory_influence']
            exitation = 'KOSKO'
            if 'exitation_function' in concept:
                exitation = concept['exitation_function']
            active = True
            if 'active' in concept:
                active = concept['active']
            custom_function = None
            if 'custom_function' in concept:
                custom_function = concept['custom_function']
            custom_func_args = {}
            if 'custom_function_args' in concept:
                custom_func_args = concept['custom_function_args']
            concept_type = TYPE_SIMPLE
            if concept['type'] == 'DECISION':
                concept_type = TYPE_DECISION
            my_fcm.add_concept(concept['id'],
                               concept_type=concept_type,
                               is_active=active,
                               use_memory=use_mem,
                               exitation_function=exitation,
                               activation_function=custom_function,
                               **custom_func_args)
            if 'activation' in concept:
                my_fcm.init_concept(concept['id'], concept['activation'])

        # adding relations
        for relation in data_dict['relations']:
            my_fcm.add_relation(origin_concept=relation['origin'],
                                destiny_concept=relation['destiny'],
                                weight=relation['weight'])
        return my_fcm
    except Exception as err:
        raise Exception("Cannot load json data due: " + str(err))


def join_maps(map_set, concept_strategy='union', value_strategy="average", relation_strategy="average",
              ignore_zeros=False):
    """
    Join a set of FuzzyCognitiveMap in a new one according to defined strategy. All nodes will be set to default
    behaviour to avid mixing issues in the result. The final map also will be created with default behavior so, is
    required to update the map behavior after join process. Default setting will be updated on future library versions.
    Args:
        map_set: An iterable object that contains the FCMs
        concept_strategy: Strategy to join all maps nodes
            union: the new FuzzyCognitiveMap will have the set union of nodes in map_set
            intersection: the new FuzzyCognitiveMap will have the set intersection of nodes in  map_set
        value_strategy: Strategy to define the initial state of map nodes
            highest: Select the highest node value as initial node state
            lowest: Select the lowest node value as initial node state
            average: Select the average of node values as initial node state
        relation_strategy: Strategy to define the value for repeated relations weight in map topology
            highest: Select the highest relation value as new relation value
            lowest: Select the lowest relation value as new relation value
            average: Select the average of relations values as new relation value
        ignore_zeros: Ignore zero evaluated concepts in value_strategy selected

    Returns: A new FuzzyCognitiveMap generated using defined strategies

    """
    concept_strategies = {'union', 'intersection'}
    value_strategies = {'highest', 'lowest', 'average'}
    relation_strategies = {'highest', 'lowest', 'average'}
    if concept_strategy not in concept_strategies:
        raise Exception("Unknown concept strategy: " + concept_strategy)
    if value_strategy not in value_strategies:
        raise Exception("Unknown value strategy: " + value_strategy)
    if relation_strategy not in relation_strategies:
        raise Exception("Unknown relation strategy: " + relation_strategy)

    nodes_desc = {}
    relations = []
    is_first = True
    final_map = {}
    for fcm in map_set:
        map_desc = json.loads(fcm.to_json())
        for relation in map_desc['relations']:
            relations.append(relation)
        if is_first:
            is_first = False
            final_map = map_desc
            for concept in map_desc['concepts']:
                nodes_desc[concept['id']] = concept
                nodes_desc[concept['id']]['accumulation'] = [nodes_desc[concept['id']]['activation']]
        else:
            new_node_set = {}
            for concept in map_desc['concepts']:
                new_node_set[concept['id']] = concept
            if concept_strategy == 'union':
                for key in new_node_set:
                    if key in nodes_desc:
                        nodes_desc[key]['accumulation'].append(new_node_set[key]['activation'])
                    else:
                        nodes_desc[key] = new_node_set[key]
                        nodes_desc[key]['accumulation'] = [nodes_desc[key]['activation']]
            if concept_strategy == 'intersection':
                node_set = set(nodes_desc.keys())
                node_set = node_set.intersection(new_node_set.keys())
                to_remove = []
                for key in nodes_desc:
                    if key not in node_set:
                        to_remove.append(key)
                    else:
                        nodes_desc[key]['accumulation'].append(new_node_set[key]['activation'])
                for key in to_remove:
                    nodes_desc.pop(key)
    final_concepts = []
    for key in nodes_desc:
        if value_strategy == "highest":
            nodes_desc[key]['activation'] = max(nodes_desc[key]['accumulation'])
        if value_strategy == "lowest":
            nodes_desc[key]['activation'] = min(nodes_desc[key]['accumulation'])
        if value_strategy == "average":
            num_elements = len(nodes_desc[key]['accumulation'])
            if num_elements > 0:
                nodes_desc[key]['activation'] = sum(nodes_desc[key]['accumulation']) / num_elements
        nodes_desc[key].pop('accumulation')
        if nodes_desc[key]['activation'] != 0:
            final_concepts.append(nodes_desc[key])
        elif not ignore_zeros:
            final_concepts.append(nodes_desc[key])

    relation_data = {}
    rel_separator = ' |=====> '
    for curr_relation in relations:
        relation_name = curr_relation['origin'] + rel_separator + curr_relation['destiny']
        if relation_name not in relation_data:
            relation_data[relation_name] = [curr_relation['weight']]
        else:
            relation_data[relation_name].append(curr_relation['weight'])

    final_relations = []
    for curr_relation in relation_data:
        origin = curr_relation.split(rel_separator)[0]
        destiny = curr_relation.split(rel_separator)[1]
        if origin in nodes_desc and destiny in nodes_desc:
            new_relation = {
                'origin': origin,
                'destiny': destiny
            }
            if relation_strategy == "highest":
                new_relation['weight'] = max(relation_data[curr_relation])
            if relation_strategy == "lowest":
                new_relation['weight'] = min(relation_data[curr_relation])
            if relation_strategy == "average":
                new_relation['weight'] = sum(relation_data[curr_relation]) / len(relation_data[curr_relation])
            final_relations.append(new_relation)

    final_map['concepts'] = final_concepts
    final_map['relations'] = final_relations
    final_json = json.dumps(final_map)
    return from_json(final_json)


class FuzzyCognitiveMap:

    def __init__(self, max_it=200, extra_steps=5, stabilize=True, stability_diff=0.001, decision_function="MEAN",
                 mem_influence=False, activ_function="sigmoid_hip", **kwargs):
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
            activ_function: Activation function for map nodes
                biestate: Dual estate function => {0,1}
                threestate: Three  estate function => {0,0.5,1}
                saturation: Values lower than 0 or highest than 1 are transformed => [0;1]
                sigmoid: Sigmoid function => [0;1]
                sigmoid_hip: Hyperbolic sigmoid function => [-1;1]
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
        self.set_map_activation_function(activ_function)

        # Set global function arguments
        self.__global_func_args = kwargs
        # map iterations
        self.__iterations = 1

        # str separator for datasets
        self.__separator = None

        # map weight for join process
        self.weight = 1
        self._default_concept = {}
        self.set_default_concept_properties()

    def set_map_decision_function(self, function_name: str):
        # Decision function definition
        new_function = Decision.get_by_name(function_name)
        if new_function is None:
            raise Exception("Unknown decision function '" + str(function_name) + "'")
        self.__decision_function = function_name
        self.__map_decision_function = new_function

    def set_map_activation_function(self, function_name: str):
        # Activation function definition
        new_function = Activation.get_by_name(function_name)
        if new_function is None:
            raise Exception("Unknown activation function '" + str(function_name) + "'")
        self.__activation_function = function_name
        self.__map_activation_function = new_function
        for concept_name in self.__topology:
            if self.__topology[concept_name][NODE_USE_MAP_FUNC]:
                self.__topology[concept_name][NODE_ACTV_FUNC] = self.__map_activation_function
                self.__topology[concept_name][NODE_ACTV_FUNC_NAME] = self.__activation_function
                self.__topology[concept_name][NODE_ACTV_FUNC_ARGS] = self.__global_func_args

    def add_concept(self, concept_name: str, concept_type=None, is_active=None, use_memory=None,
                    exitation_function=None, activation_dict=None, activation_function=None, **kwargs):
        """
        Add new concept to map
        Args:
            concept_name: Concept name
            concept_type: Define type of node and behavior
            is_active: Define if node is active or not
            use_memory: Use memory in activation node process
            exitation_function: Custom function name for execution process
            activation_function: Function name for node activation, if none set default defined function
            activation_dict: activation dic for cont concepts according to found clusters in way =>
                {'membership': [], 'val_list': []} and related by position.
            **kwargs: arguments for activation function

        Returns: None

        Exitation functions:
        * "MEAN": Mean values of all neighbors that influence the node
        * "KOSKO": B. Kosko proposed activation function
        * "PAPAGEORGIUS": E. Papageorgius proposed function to avoid saturation

        Activation functions:
        * "saturation": 1 if value is > 1, 0 if values is < 0 and value otherwise. Domain => [0,1]
        * "biestate": 1 if value is > 0, 0 otherwise. Domain => {0,1}
        * "threestate": 0 if value is < 0.25, 0.5 if 0.25 <= value <= 0.75, 1 otherwise. Domain => {0,0.5,1}
        * "sum_w": weight(float), return value if > weight, 0 otherwise. Domain => [-1,1]
        * "sigmoid": lambda_val(int), sigmoid function => [0,1]
        * "sigmoid_hip": lambda_val(int), sigmoid hyperbolic function => [-1,1]
        """
        self.__topology[concept_name] = {NODE_ARCS: [], NODE_AUX: [], NODE_VALUE: 0.0}

        if is_active is not None and type(is_active) == bool:
            self.__topology[concept_name][NODE_ACTIVE] = is_active
        else:
            self.__topology[concept_name][NODE_ACTIVE] = self._default_concept[NODE_ACTIVE]

        if exitation_function is not None:
            function = Excitation.get_by_name(exitation_function)
            if function is not None:
                self.__topology[concept_name][NODE_EXEC_FUNC] = function
                self.__topology[concept_name][NODE_EXEC_FUNC_NAME] = exitation_function
            else:
                raise Exception("Unknown exitation function: " + exitation_function)
        else:
            self.__topology[concept_name][NODE_EXEC_FUNC] = self._default_concept[NODE_EXEC_FUNC]
            self.__topology[concept_name][NODE_EXEC_FUNC_NAME] = self._default_concept[NODE_EXEC_FUNC_NAME]

        if is_valid_type(concept_type) and concept_type is not None:
            self.__topology[concept_name][NODE_TYPE] = concept_type
        else:
            self.__topology[concept_name][NODE_TYPE] = self._default_concept[NODE_TYPE]

        # scale and normalize the values for fuzzy function
        # activation_dict = {'membership':[],'val_list':[]}
        if (concept_type == TYPE_FUZZY or concept_type == TYPE_REGRESOR) and activation_dict is not None:
            self.__topology[concept_name][NODE_EXEC_FUNC] = Excitation.get_by_name('MEAN')
            self.__topology[concept_name][NODE_EXEC_FUNC_NAME] = 'MEAN'
            self.__topology[concept_name][NODE_TRAIN_ACTIVATION] = activation_dict
            # scale for only positive values
            self.__topology[concept_name][NODE_TRAIN_MIN] = abs(min(activation_dict['val_list']))
            for pos in range(len(activation_dict['val_list'])):
                activation_dict['val_list'][pos] += self.__topology[concept_name][NODE_TRAIN_MIN]
            # normalize positive values
            self.__topology[concept_name][NODE_TRAIN_FMAX] = max(activation_dict['val_list'])
            for pos in range(len(activation_dict['val_list'])):
                activation_dict['val_list'][pos] = activation_dict['val_list'][pos] / self.__topology[concept_name][
                    NODE_TRAIN_FMAX]

        if use_memory is not None and type(use_memory) == bool:
            self.__topology[concept_name][NODE_USE_MEM] = use_memory
        else:
            self.__topology[concept_name][NODE_USE_MEM] = self._default_concept[NODE_USE_MEM]

        # define activation function
        self.__topology[concept_name][NODE_USE_MAP_FUNC] = False
        if activation_dict is not None:
            self.__topology[concept_name][NODE_ACTV_FUNC] = Activation.fuzzy_set
            self.__topology[concept_name][NODE_ACTV_FUNC_NAME] = 'FUZZY'
            self.__topology[concept_name][NODE_ACTV_FUNC_ARGS] = activation_dict
        elif activation_function is not None:
            actv_function = Activation.get_by_name(activation_function)
            if actv_function is not None:
                self.__topology[concept_name][NODE_ACTV_FUNC] = actv_function
                self.__topology[concept_name][NODE_ACTV_FUNC_NAME] = activation_function
                self.__topology[concept_name][NODE_ACTV_FUNC_ARGS] = kwargs
            else:
                raise Exception("Unknown ativation function: " + activation_function)
        else:
            self.__topology[concept_name][NODE_ACTV_FUNC] = self._default_concept[NODE_ACTV_FUNC]
            self.__topology[concept_name][NODE_ACTV_FUNC_NAME] = self._default_concept[NODE_ACTV_FUNC_NAME]
            self.__topology[concept_name][NODE_ACTV_FUNC_ARGS] = self._default_concept[NODE_ACTV_FUNC_ARGS]
            self.__topology[concept_name][NODE_USE_MAP_FUNC] = True

        self.__execution[concept_name] = [0.0]
        self.__topology[concept_name][NODE_VALUE] = 0.0

    def set_concept_properties(self, concept_name: str, concept_type=None, is_active=None, use_memory=None,
                               exitation_function=None, activation_dict=None, activation_function=None, **kwargs):
        if concept_name in self.__topology:
            if is_active is not None and type(is_active) == bool:
                self.__topology[concept_name][NODE_ACTIVE] = is_active
            if exitation_function is not None:
                new_function = Excitation.get_by_name(exitation_function)
                if new_function is not None:
                    self.__topology[concept_name][NODE_EXEC_FUNC] = new_function
                    self.__topology[concept_name][NODE_EXEC_FUNC_NAME] = exitation_function
                else:
                    raise Exception("Unknown exitation function: " + exitation_function)
            if concept_type is not None:
                if is_valid_type(concept_type):
                    self.__topology[concept_name][NODE_TYPE] = concept_type
                else:
                    raise Exception("Unknown node type")

            # scale and normalize the values for fuzzy function
            # activation_dict = {'membership':[],'val_list':[]}
            if (concept_type == TYPE_FUZZY or concept_type == TYPE_REGRESOR) and activation_dict is not None:
                self.__topology[concept_name][NODE_EXEC_FUNC] = Excitation.get_by_name('MEAN')
                self.__topology[concept_name][NODE_EXEC_FUNC_NAME] = 'MEAN'
                self.__topology[concept_name][NODE_TRAIN_ACTIVATION] = activation_dict
                # scale for only positive values
                self.__topology[concept_name][NODE_TRAIN_MIN] = abs(min(activation_dict['val_list']))
                for pos in range(len(activation_dict['val_list'])):
                    activation_dict['val_list'][pos] += self.__topology[concept_name][NODE_TRAIN_MIN]
                # normalize positive values
                self.__topology[concept_name][NODE_TRAIN_FMAX] = max(activation_dict['val_list'])
                for pos in range(len(activation_dict['val_list'])):
                    activation_dict['val_list'][pos] = activation_dict['val_list'][pos] / self.__topology[concept_name][
                        NODE_TRAIN_FMAX]

            if use_memory is not None and type(use_memory) == bool:
                self.__topology[concept_name][NODE_USE_MEM] = use_memory
            else:
                self.__topology[concept_name][NODE_USE_MEM] = self.flag_mem_influence

            # define activation function
            self.__topology[concept_name][NODE_USE_MAP_FUNC] = False
            if activation_dict is not None:
                self.__topology[concept_name][NODE_ACTV_FUNC] = Activation.fuzzy_set
                self.__topology[concept_name][NODE_ACTV_FUNC_NAME] = 'FUZZY'
                self.__topology[concept_name][NODE_ACTV_FUNC_ARGS] = activation_dict
            elif activation_function is not None:
                actv_function = Activation.get_by_name(activation_function)
                if actv_function is not None:
                    self.__topology[concept_name][NODE_ACTV_FUNC] = actv_function
                    self.__topology[concept_name][NODE_ACTV_FUNC_NAME] = activation_function
                    self.__topology[concept_name][NODE_ACTV_FUNC_ARGS] = kwargs
                else:
                    raise Exception("Unknown ativation function: " + activation_function)
            else:
                self.__topology[concept_name][NODE_ACTV_FUNC] = self.__map_activation_function
                self.__topology[concept_name][NODE_ACTV_FUNC_NAME] = self.__activation_function
                self.__topology[concept_name][NODE_ACTV_FUNC_ARGS] = self.__global_func_args
                self.__topology[concept_name][NODE_USE_MAP_FUNC] = True
        else:
            raise Exception("Concept " + concept_name + " not found")

    def set_default_concept_properties(self, concept_type=TYPE_SIMPLE, is_active=True, use_memory=None,
                                       exitation_function='KOSKO', activation_dict=None, activation_function=None,
                                       **kwargs):
        if is_active is not None and type(is_active) == bool:
            self._default_concept[NODE_ACTIVE] = is_active

        new_function = Excitation.get_by_name(exitation_function)
        if new_function is not None:
            self._default_concept[NODE_EXEC_FUNC] = new_function
            self._default_concept[NODE_EXEC_FUNC_NAME] = exitation_function
        else:
            raise Exception("Unknown exitation function: " + exitation_function)
        if is_valid_type(concept_type):
            self._default_concept[NODE_TYPE] = concept_type
        else:
            raise Exception("Unknown node type")

        if concept_type == TYPE_FUZZY or concept_type == TYPE_REGRESOR and activation_dict is not None:
            self._default_concept[NODE_EXEC_FUNC] = Excitation.get_by_name('MEAN')
            self._default_concept[NODE_EXEC_FUNC_NAME] = 'MEAN'
            self._default_concept[NODE_TRAIN_ACTIVATION] = activation_dict
            # scale for only positive values
            self._default_concept[NODE_TRAIN_MIN] = abs(min(activation_dict['val_list']))
            for pos in range(len(activation_dict['val_list'])):
                activation_dict['val_list'][pos] += self._default_concept[NODE_TRAIN_MIN]
            # normalize positive values
            self._default_concept[NODE_TRAIN_FMAX] = max(activation_dict['val_list'])
            for pos in range(len(activation_dict['val_list'])):
                activation_dict['val_list'][pos] = activation_dict['val_list'][pos] / self._default_concept[
                    NODE_TRAIN_FMAX]

        if use_memory is not None and type(use_memory) == bool:
            self._default_concept[NODE_USE_MEM] = use_memory
        else:
            self._default_concept[NODE_USE_MEM] = self.flag_mem_influence

        # define activation function
        self._default_concept[NODE_USE_MAP_FUNC] = False
        if activation_dict is not None:
            self._default_concept[NODE_ACTV_FUNC] = Activation.fuzzy_set
            self._default_concept[NODE_ACTV_FUNC_NAME] = 'FUZZY'
            self._default_concept[NODE_ACTV_FUNC_ARGS] = activation_dict
        elif activation_function is not None:
            actv_function = Activation.get_by_name(activation_function)
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
        if destiny_concept in self.__topology and origin_concept in self.__topology:
            self.__topology[origin_concept][NODE_ARCS].append((destiny_concept, weight))
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

    def init_concept(self, concept_name: str, value: float, required_presence=True):
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
        self.__topology[concept_name][NODE_VALUE] = value
        self.__execution[concept_name] = [value]

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
        for node, data in self.__topology.items():
            data[NODE_AUX] = []
            data[NODE_VALUE] = self.__execution[node][0]
            self.__execution[node] = [data[NODE_VALUE]]

    def clear_execution(self):
        """
        Clear map execution and inital concepts values for a new inference process
        Returns: None

        """
        for node, data in self.__topology.items():
            data[NODE_AUX] = []
            data[NODE_VALUE] = 0.0
            self.__execution[node] = [0.0]

    def run_inference(self, reset=True):
        """
        Execute map inference process
        Returns: None

        """
        if reset:
            self.reset_execution()
        self.__iterations = 1
        extra_steps = self.__extra_steps
        while extra_steps > 0:
            # execute extra_steps new iterations after finish execution
            if not self.__keep_execution():
                extra_steps -= 1
            for arc in self.__arc_list:
                origin = arc[RELATION_ORIGIN]
                dest = arc[RELATION_DESTINY]
                weight = arc[RELATION_WEIGHT]
                # set value to: sum(wij * Ai)
                if self.__topology[origin][NODE_ACTIVE]:
                    self.__topology[dest][NODE_AUX].append(self.__topology[origin][NODE_VALUE] * weight)

            for node in self.__execution.keys():
                # calc execution value using NODE_AUX, NODE_VALUE, NODE_USE_MEM and NODE_ACTIVE, the values may change
                exec_val = self.__topology[node][NODE_EXEC_FUNC](self.__topology[node])
                # reset NODE_AUX after calculate the execution value
                self.__topology[node][NODE_AUX].clear()  # = []
                result = self.__topology[node][NODE_ACTV_FUNC](exec_val, **self.__topology[node][NODE_ACTV_FUNC_ARGS])
                # update execution values
                self.__topology[node][NODE_VALUE] = result
                self.__execution[node].append(result)

            self.__iterations += 1

        for node in self.__topology:
            self.__topology[node][NODE_VALUE] = self.__map_decision_function(self.__execution[node])

    def get_final_state(self, concepts_type="target", names=[]):
        """
        Get inference result values of node_type or all nodes
        Args:
            concepts_type: Type of nodes inference result
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
                                                            self.__topology[concept_name][NODE_TRAIN_MIN],
                                                            self.__topology[concept_name][NODE_TRAIN_FMAX],
                                                            **self.__topology[concept_name][NODE_TRAIN_ACTIVATION]))
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
                                                           self.__topology[concept][NODE_TRAIN_MIN],
                                                           self.__topology[concept][NODE_TRAIN_FMAX],
                                                           **self.__topology[concept][NODE_TRAIN_ACTIVATION])
                    self.__topology[concept][NODE_VALUE] = activation_value
                    self.__execution[concept] = [activation_value]
                processed = True
            if not processed:
                raise Exception(
                    "Error processing value " + str(value) + " from feature '" + feature + "' and type '" + str(
                        type(value)) + "'")
        else:
            raise Warning("Concepts related to feature '" + feature + "' are not defined in FCM topology.")
