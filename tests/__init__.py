from py_fcm.utils.__const import *
from py_fcm.utils.functions import Excitation, Activation


def create_concept(node_type=TYPE_SIMPLE, is_active=True, use_memory=True,
                   exitation_function='KOSKO', activ_function=None, **kwargs):
    test_concept = {NODE_ACTIVE: is_active, NODE_ARCS: [], NODE_AUX: [1, 1, 1], NODE_VALUE: 1.0}
    test_concept[NODE_TYPE] = node_type
    test_concept[NODE_EXEC_FUNC] = Excitation.get_by_name(exitation_function)
    test_concept[NODE_USE_MEM] = use_memory
    test_concept[NODE_ACTV_FUNC] = Activation.get_function_by_name(activ_function)
    test_concept[NODE_ACTV_FUNC_ARGS] = kwargs
    return test_concept
