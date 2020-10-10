# None activation, transmit input values to connections
FUNC_DIRECT = 0
# Sigmoid-type function
FUNC_SIGMOID = 1
# Saturation-type function
FUNC_SATURATION = 2
# Tristate-type function
FUNC_TRISTATE = 3
# Bistate-type function
FUNC_BISTATE = 4
# Fuzzy-type function
FUNC_FUZZY = 5
# Pulse-type function
FUNC_PULSE = 6
# Stable mean behavior
FUNC_MEAN_GAP = 7

# relations const
RELATION_DESTINY = 0
RELATION_WEIGHT = 1
RELATION_ORIGIN = 2

# topology const
NODE_ACTIVE = "current node is active or not"  # non active nodes do not activate neighbors
NODE_ARCS = "current node outgoing arcs"
NODE_TYPE = "type of node"
NODE_VALUE = "current node value"
NODE_USE_MEM = "use memory in current node execution"
NODE_AUX = "auxiliary influence value list for new value calculation"
NODE_USE_MAP_FUNC = "use map activation function"
NODE_EXEC_FUNC = "custom execution function"
NODE_EXEC_FUNC_NAME = "custom execution function name"
NODE_ACTV_FUNC = "custom activation function"
NODE_ACTV_FUNC_NAME = "custom activation function name"
NODE_ACTV_FUNC_ARGS = "custom activation function arguments"
NODE_TRAIN_ACTIVATION = "activation relation for continuous values"
NODE_TRAIN_MIN = "minimum continuous value"
NODE_TRAIN_FMAX = "minimum continuous value + abs(NODE_TRAIN_MIN)"

# node types
TYPE_SIMPLE = "default node type"
TYPE_DECISION = "node type for classification problems"
TYPE_FUZZY = "node type for fuzzy treatment of continuous values"
TYPE_REGRESOR = "node type for regression problems"
TYPE_MUTI = "node type for multivalues fields"
TYPE_MUTI_DESC = "node type for multivalues fields in decision nodes"


def is_valid_type(node_type) -> bool:
    type_set = set()
    type_set.add(TYPE_SIMPLE)
    type_set.add(TYPE_DECISION)
    type_set.add(TYPE_FUZZY)
    type_set.add(TYPE_REGRESOR)
    type_set.add(TYPE_MUTI)
    type_set.add(TYPE_MUTI_DESC)
    return node_type in type_set
