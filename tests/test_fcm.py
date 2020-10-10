import unittest
import json
from py_fcm import FuzzyCognitiveMap, TYPE_DECISION


class FuzzyCognitiveMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fcm = FuzzyCognitiveMap()

    def __init_complex_fcm(self):
        self.fcm = FuzzyCognitiveMap()

        self.fcm.add_concept('result_1', concept_type=TYPE_DECISION)
        self.fcm.add_concept('result_2', concept_type=TYPE_DECISION)

        self.fcm.add_concept('input_1')
        self.fcm.init_concept('input_1', 0.5)
        self.fcm.add_concept('input_2')
        self.fcm.init_concept('input_2', 0.2)
        self.fcm.add_concept('input_3')
        self.fcm.init_concept('input_3', 1)
        self.fcm.add_concept('input_4')
        self.fcm.init_concept('input_4', -0.2)
        self.fcm.add_concept('input_5')
        self.fcm.init_concept('input_5', -0.5)

        # self.fcm.add_relation('result_1', 'result_2', -1)
        # self.fcm.add_relation('result_2', 'result_1', -1)

        self.fcm.add_relation('input_1', 'result_1', 0.5)
        self.fcm.add_relation('input_2', 'result_1', 1)

        self.fcm.add_relation('input_4', 'result_2', 0.5)
        self.fcm.add_relation('input_5', 'result_2', 1)

        self.fcm.add_relation('input_1', 'input_4', -0.3)
        self.fcm.add_relation('input_1', 'input_3', 0.7)
        self.fcm.add_relation('input_5', 'input_2', -0.3)
        self.fcm.add_relation('input_5', 'input_3', 0.7)

        # self.fcm.add_relation('input_3', 'input_2', -0.1)
        # self.fcm.add_relation('input_3', 'input_4', -0.1)
        self.fcm.add_relation('result_2', 'input_3', 0.5)
        self.fcm.add_relation('result_1', 'input_3', 0.5)

    def test_default_and_to_json(self) -> None:
        expected_json = {
            "max_iter": 200,
            "decision_function": "MEAN",
            "activation_function": "sigmoid_hip",
            "memory_influence": False,
            "stability_diff": 0.001,
            "stop_at_stabilize": True,
            "extra_steps": 5,
            "weight": 1,
            "concepts": [],
            "relations": []
        }
        json_fcm = json.loads(self.fcm.to_json())
        self.assertEqual(expected_json, json_fcm)

    def test_set_map_fuctions(self) -> None:
        expected_json = {
            "max_iter": 200,
            "decision_function": "LAST",
            "activation_function": "sigmoid",
            "memory_influence": False,
            "stability_diff": 0.001,
            "stop_at_stabilize": True,
            "extra_steps": 5,
            "weight": 1,
            "concepts": [],
            "relations": []
        }
        self.fcm.set_map_decision_function('LAST')
        self.fcm.set_map_activation_function('sigmoid')
        json_fcm = json.loads(self.fcm.to_json())
        self.assertEqual(expected_json, json_fcm)

    def test_concept_addition_default(self) -> None:
        expected_json = {
            "concepts": [{'id': 'test', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.0}]
        }
        self.fcm.add_concept('test')
        json_fcm = json.loads(self.fcm.to_json())
        self.assertEqual(expected_json["concepts"], json_fcm["concepts"])

    def test_concept_init_and_get_value(self) -> None:
        self.fcm.add_concept('test')
        self.fcm.init_concept('test', 0.5)
        self.assertEqual(0.5, self.fcm.get_concept_value('test'))

    def test_concept_redefinition(self) -> None:
        expected_json = {
            "concepts": [{'id': 'test', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.0}]
        }
        self.fcm.add_concept('test')
        json_fcm = json.loads(self.fcm.to_json())
        self.assertEqual(expected_json["concepts"], json_fcm["concepts"])
        expected_json["concepts"][0]['type'] = 'DECISION'

        self.fcm.add_concept('test', concept_type=TYPE_DECISION)
        json_fcm = json.loads(self.fcm.to_json())
        self.assertEqual(expected_json["concepts"], json_fcm["concepts"])

    def test_concept_addition_custom_values(self) -> None:
        expected_json = {
            "concepts": [{'id': 'test', 'is_active': False, 'type': 'DECISION', 'activation': 0.0, 'use_memory': True,
                          'custom_function': 'sum_w', 'custom_function_args': {'weight': 0.3}}]
        }
        self.fcm.add_concept('test', is_active=False, concept_type=TYPE_DECISION, use_memory=True,
                             exitation_function='PAPAGEORGIUS', activation_function='sum_w', weight=0.3)
        json_fcm = json.loads(self.fcm.to_json())
        self.assertEqual(expected_json["concepts"], json_fcm["concepts"])

    def test_concept_addition_with_default_definition(self) -> None:
        expected_json = {
            "concepts": [{'id': 'test', 'is_active': True, 'type': 'DECISION', 'activation': 0.0, 'use_memory': True,
                          'custom_function': 'sum_w', 'custom_function_args': {'weight': 0.3}}]
        }
        self.fcm.set_default_concept_properties(concept_type=TYPE_DECISION, use_memory=True,
                                                exitation_function='PAPAGEORGIUS', activation_function='sum_w',
                                                weight=0.3)
        self.fcm.add_concept('test')
        json_fcm = json.loads(self.fcm.to_json())
        self.assertEqual(expected_json["concepts"], json_fcm["concepts"])

    def test_concept_property_update(self) -> None:
        expected_json = {
            "concepts": [{'id': 'test', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.0}]
        }
        self.fcm.add_concept('test')
        json_fcm = json.loads(self.fcm.to_json())
        self.assertEqual(expected_json["concepts"], json_fcm["concepts"])
        self.fcm.set_concept_properties('test', is_active=False, concept_type=TYPE_DECISION, use_memory=True,
                                        exitation_function='PAPAGEORGIUS', activation_function='sum_w', weight=0.3)
        expected_json = {
            "concepts": [{'id': 'test', 'is_active': False, 'type': 'DECISION', 'activation': 0.0, 'use_memory': True,
                          'custom_function': 'sum_w', 'custom_function_args': {'weight': 0.3}}]
        }
        json_fcm = json.loads(self.fcm.to_json())
        self.assertEqual(expected_json["concepts"], json_fcm["concepts"])

    def test_relation_addition(self) -> None:
        expected_json = {
            "concepts": [{'id': 'test', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.0},
                         {'id': 'test_1', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.0}],
            "relations": [{'origin': 'test', 'destiny': 'test_1', 'weight': 0.5}]
        }
        self.fcm.add_concept('test')
        self.fcm.add_concept('test_1')
        self.fcm.add_relation('test', 'test_1', 0.5)
        json_fcm = json.loads(self.fcm.to_json())
        self.assertEqual(expected_json["relations"], json_fcm["relations"])

    def test_clear_all(self) -> None:
        expected_json = {
            "max_iter": 200,
            "decision_function": "MEAN",
            "activation_function": "sigmoid_hip",
            "memory_influence": False,
            "stability_diff": 0.001,
            "stop_at_stabilize": True,
            "extra_steps": 5,
            "weight": 1,
            "concepts": [],
            "relations": []
        }
        self.fcm.add_concept('test')
        self.fcm.add_concept('test_1')
        self.fcm.add_relation('test', 'test_1', 0.5)
        self.fcm.clear_all()
        json_fcm = json.loads(self.fcm.to_json())
        self.assertEqual(expected_json, json_fcm)

    def test_inference_default(self):
        # sigmoid_hip
        expected_json = {
            'concepts': [{'id': 'result_1', 'is_active': True, 'type': 'DECISION', 'activation': 0.051790327919945374},
                         {'id': 'result_2', 'is_active': True, 'type': 'DECISION', 'activation': -0.055577715958134676},
                         {'id': 'input_1', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.045454545454545456},
                         {'id': 'input_2', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.031716821238483454},
                         {'id': 'input_3', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.08901967021146369},
                         {'id': 'input_4', 'is_active': True, 'type': 'SIMPLE', 'activation': -0.031716821238483454},
                         {'id': 'input_5', 'is_active': True, 'type': 'SIMPLE', 'activation': -0.045454545454545456}]
        }

        self.__init_complex_fcm()
        self.fcm.run_inference()
        json_fcm = json.loads(self.fcm.to_json())
        self.assertEqual(expected_json["concepts"], json_fcm["concepts"])

    def test_is_stable(self):
        self.__init_complex_fcm()
        self.fcm.run_inference()
        self.assertEqual(True, self.fcm.is_stable())

    def test_get_final_state_default(self):
        self.__init_complex_fcm()
        self.fcm.run_inference()
        expected_result = {'result_1': 0.051790327919945374, 'result_2': -0.055577715958134676}
        self.assertEqual(expected_result, self.fcm.get_final_state())

    def test_search_concept_final_state_arg(self):
        self.__init_complex_fcm()
        self.fcm.run_inference()
        expected_result = {'result_1': 0.051790327919945374}
        self.assertEqual(expected_result, self.fcm.get_final_state(names=['result_1']))

    def test_get_final_state_any(self):
        self.__init_complex_fcm()
        self.fcm.run_inference()
        expected_result = {'result_1': 0.051790327919945374, 'result_2': -0.055577715958134676,
                           'input_1': 0.045454545454545456, 'input_2': 0.031716821238483454,
                           'input_3': 0.08901967021146369, 'input_4': -0.031716821238483454,
                           'input_5': -0.045454545454545456}
        self.assertEqual(expected_result, self.fcm.get_final_state("any"))

    def test_get_final_state_custom_type(self):
        self.__init_complex_fcm()
        self.fcm.run_inference()
        expected_result = {'result_1': 0.051790327919945374, 'result_2': -0.055577715958134676}
        self.assertEqual(expected_result, self.fcm.get_final_state(TYPE_DECISION))

    def test_reset_execution(self):
        self.__init_complex_fcm()
        self.fcm.run_inference()
        self.fcm.reset_execution()
        expected_result = {'input_1': 0.5}
        self.assertEqual(expected_result, self.fcm.get_final_state(names=['input_1']))

    def test_clear_execution(self):
        self.__init_complex_fcm()
        self.fcm.run_inference()
        self.fcm.clear_execution()
        expected_result = {'input_1': 0.0}
        self.assertEqual(expected_result, self.fcm.get_final_state(names=['input_1']))

    def test_inference_sigmoid(self):
        expected_json = {
            'concepts': [{'id': 'result_1', 'is_active': True, 'type': 'DECISION', 'activation': 0.6059699095582007},
                         {'id': 'result_2', 'is_active': True, 'type': 'DECISION', 'activation': 0.5845571444568803},
                         {'id': 'input_1', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.5},
                         {'id': 'input_2', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.44550556702272764},
                         {'id': 'input_3', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.7863326348226287},
                         {'id': 'input_4', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.402336504232955},
                         {'id': 'input_5', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.4090909090909091}]
        }

        self.__init_complex_fcm()
        self.fcm.set_map_activation_function('sigmoid')
        self.fcm.run_inference()
        json_fcm = json.loads(self.fcm.to_json())
        self.assertEqual(expected_json["concepts"], json_fcm["concepts"])

    def test_inference_biestate(self):
        expected_json = {
            'concepts': [{'id': 'result_1', 'is_active': True, 'type': 'DECISION', 'activation': 0.18181818181818182},
                         {'id': 'result_2', 'is_active': True, 'type': 'DECISION', 'activation': 0.0},
                         {'id': 'input_1', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.045454545454545456},
                         {'id': 'input_2', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.10909090909090909},
                         {'id': 'input_3', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.2727272727272727},
                         {'id': 'input_4', 'is_active': True, 'type': 'SIMPLE', 'activation': -0.018181818181818184},
                         {'id': 'input_5', 'is_active': True, 'type': 'SIMPLE', 'activation': -0.045454545454545456}]
        }

        self.__init_complex_fcm()
        self.fcm.set_map_activation_function('biestate')
        self.fcm.run_inference()
        json_fcm = json.loads(self.fcm.to_json())
        self.assertEqual(expected_json["concepts"], json_fcm["concepts"])

    def test_inference_threestate(self):
        expected_json = {
            'concepts': [{'id': 'result_1', 'is_active': True, 'type': 'DECISION', 'activation': 0.05555555555555555},
                         {'id': 'result_2', 'is_active': True, 'type': 'DECISION', 'activation': 0.0},
                         {'id': 'input_1', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.05555555555555555},
                         {'id': 'input_2', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.022222222222222223},
                         {'id': 'input_3', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.1111111111111111},
                         {'id': 'input_4', 'is_active': True, 'type': 'SIMPLE', 'activation': -0.022222222222222223},
                         {'id': 'input_5', 'is_active': True, 'type': 'SIMPLE', 'activation': -0.05555555555555555}]
        }

        self.__init_complex_fcm()
        self.fcm.set_map_activation_function('threestate')
        self.fcm.run_inference()
        json_fcm = json.loads(self.fcm.to_json())
        self.assertEqual(expected_json["concepts"], json_fcm["concepts"])

    def test_inference_saturation(self):
        expected_json = {
            'concepts': [{'id': 'result_1', 'is_active': True, 'type': 'DECISION', 'activation': 0.05454545454545454},
                         {'id': 'result_2', 'is_active': True, 'type': 'DECISION', 'activation': 0.0},
                         {'id': 'input_1', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.045454545454545456},
                         {'id': 'input_2', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.031818181818181815},
                         {'id': 'input_3', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.11818181818181818},
                         {'id': 'input_4', 'is_active': True, 'type': 'SIMPLE', 'activation': -0.018181818181818184},
                         {'id': 'input_5', 'is_active': True, 'type': 'SIMPLE', 'activation': -0.045454545454545456}]
        }

        self.__init_complex_fcm()
        self.fcm.set_map_activation_function('saturation')
        self.fcm.run_inference()
        json_fcm = json.loads(self.fcm.to_json())
        self.assertEqual(expected_json["concepts"], json_fcm["concepts"])
