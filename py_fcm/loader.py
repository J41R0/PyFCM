import json

from py_fcm.utils.__const import *
from py_fcm.fcm import FuzzyCognitiveMap


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
        {"id": "concept_2", "type": "DECISION", "custom_function": "gceq", "custom_function_args": {"weight":0.3}},
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
    * "gceq": weight(float), return value if > weight, 0 otherwise. Domain => [-1,1]
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
                                   activation_function=data_dict['activation_function'],
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
            activation_dict = None
            if 'activation_dict' in concept:
                activation_dict = concept['activation_dict']
            concept_type = TYPE_SIMPLE
            if concept['type'] == 'DECISION':
                concept_type = TYPE_DECISION
            my_fcm.add_concept(concept['id'],
                               concept_type=concept_type,
                               is_active=active,
                               use_memory=use_mem,
                               excitation_function=exitation,
                               activation_function=custom_function,
                               activation_dict=activation_dict,
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
    if len(map_set) > 0:
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
    return FuzzyCognitiveMap()
