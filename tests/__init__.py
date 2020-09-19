from py_fcm.__const import *
from py_fcm.functions import Excitation


def create_concept(concept_name: str, node_type=TYPE_SIMPLE, is_active=True, use_memory=None,
                   exitation_function='KOSKO', activation_dict=None, activ_function=None, **kwargs):
    test_concept = {NODE_ACTIVE: is_active, NODE_ARCS: [], NODE_AUX: [], NODE_VALUE: 0.0}
    test_concept[NODE_TYPE] = node_type
    test_concept[NODE_EXEC_FUNC] = Excitation.get_by_name(exitation_function)
