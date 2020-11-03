import unittest
import json
from py_fcm import FuzzyCognitiveMap, TYPE_DECISION


class FuzzyCognitiveMapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fcm = FuzzyCognitiveMap()

    def __init_complex_fcm(self):
        fcm = FuzzyCognitiveMap()

        fcm.add_concept('result_1', concept_type=TYPE_DECISION)
        fcm.add_concept('result_2', concept_type=TYPE_DECISION)

        fcm.add_concept('input_1')
        fcm.init_concept('input_1', 0.5)
        fcm.add_concept('input_2')
        fcm.init_concept('input_2', 0.2)
        fcm.add_concept('input_3')
        fcm.init_concept('input_3', 1)
        fcm.add_concept('input_4')
        fcm.init_concept('input_4', -0.2)
        fcm.add_concept('input_5')
        fcm.init_concept('input_5', -0.5)

        fcm.add_relation('input_1', 'result_1', 0.5)
        fcm.add_relation('input_2', 'result_1', 1)

        fcm.add_relation('input_4', 'result_2', 0.5)
        fcm.add_relation('input_5', 'result_2', 1)

        fcm.add_relation('input_1', 'input_4', -0.3)
        fcm.add_relation('input_1', 'input_3', 0.7)
        fcm.add_relation('input_5', 'input_2', -0.3)
        fcm.add_relation('input_5', 'input_3', 0.7)

        fcm.add_relation('result_2', 'input_3', 0.5)
        fcm.add_relation('result_1', 'input_3', 0.5)
        return fcm

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
            'activation_function_args': {'lambda_val': 10},
            "memory_influence": False,
            "stability_diff": 0.001,
            "stop_at_stabilize": True,
            "extra_steps": 5,
            "weight": 1,
            "concepts": [],
            "relations": []
        }
        self.fcm.set_map_decision_function('LAST')
        self.fcm.set_map_activation_function('sigmoid', lambda_val=10)
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
            'concepts': [{'id': 'result_1', 'is_active': True, 'type': 'DECISION', 'activation': 0.056969360711939906},
                         {'id': 'result_2', 'is_active': True, 'type': 'DECISION', 'activation': -0.06113548755394814},
                         {'id': 'input_1', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.05},
                         {'id': 'input_2', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.034888503362331805},
                         {'id': 'input_3', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.09792163723261006},
                         {'id': 'input_4', 'is_active': True, 'type': 'SIMPLE', 'activation': -0.034888503362331805},
                         {'id': 'input_5', 'is_active': True, 'type': 'SIMPLE', 'activation': -0.05}]
        }

        fcm = self.__init_complex_fcm()
        fcm.run_inference()
        json_fcm = json.loads(fcm.to_json())
        self.assertEqual(expected_json["concepts"], json_fcm["concepts"])

    def test_is_stable(self):
        fcm = self.__init_complex_fcm()
        fcm.run_inference()
        self.assertEqual(True, fcm.is_stable())

    def test_get_final_state_default(self):
        fcm = self.__init_complex_fcm()
        fcm.run_inference()
        expected_result = {'result_1': 0.056969360711939906, 'result_2': -0.06113548755394814}
        self.assertEqual(expected_result, fcm.get_final_state())

    def test_search_concept_final_state_arg(self):
        fcm = self.__init_complex_fcm()
        fcm.run_inference()
        expected_result = {'result_1': 0.056969360711939906}
        self.assertEqual(expected_result, fcm.get_final_state(names=['result_1']))

    def test_get_final_state_any(self):
        fcm = self.__init_complex_fcm()
        fcm.run_inference()
        expected_result = {'result_1': 0.056969360711939906, 'result_2': -0.06113548755394814,
                           'input_1': 0.05, 'input_2': 0.034888503362331805,
                           'input_3': 0.09792163723261006, 'input_4': -0.034888503362331805,
                           'input_5': -0.05}
        self.assertEqual(expected_result, fcm.get_final_state("any"))

    def test_get_final_state_custom_type(self):
        fcm = self.__init_complex_fcm()
        fcm.run_inference()
        expected_result = {'result_1': 0.056969360711939906, 'result_2': -0.06113548755394814}
        self.assertEqual(expected_result, fcm.get_final_state(TYPE_DECISION))

    def test_reset_execution(self):
        fcm = self.__init_complex_fcm()
        fcm.run_inference()
        fcm.reset_execution()
        expected_result = {'input_1': 0.5}
        self.assertEqual(expected_result, fcm.get_final_state(names=['input_1']))

    def test_clear_execution(self):
        fcm = self.__init_complex_fcm()
        fcm.run_inference()
        fcm.clear_execution()
        expected_result = {'input_1': 0.0}
        self.assertEqual(expected_result, fcm.get_final_state(names=['input_1']))

    def test_inference_sigmoid(self):
        expected_json = {
            'concepts': [{'id': 'result_1', 'is_active': True, 'type': 'DECISION', 'activation': 0.5994700184028905},
                         {'id': 'result_2', 'is_active': True, 'type': 'DECISION', 'activation': 0.5755041378442186},
                         {'id': 'input_1', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.5},
                         {'id': 'input_2', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.44379910825937535},
                         {'id': 'input_3', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.7851790048472618},
                         {'id': 'input_4', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.3963131391906254},
                         {'id': 'input_5', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.4}]
        }

        fcm = self.__init_complex_fcm()
        fcm.set_map_activation_function('sigmoid')
        fcm.run_inference()
        json_fcm = json.loads(fcm.to_json())
        self.assertEqual(expected_json["concepts"], json_fcm["concepts"])

    def test_inference_biestate(self):
        expected_json = {
            'concepts': [{'id': 'result_1', 'is_active': True, 'type': 'DECISION', 'activation': 0.2},
                         {'id': 'result_2', 'is_active': True, 'type': 'DECISION', 'activation': 0.0},
                         {'id': 'input_1', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.05},
                         {'id': 'input_2', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.12},
                         {'id': 'input_3', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.3},
                         {'id': 'input_4', 'is_active': True, 'type': 'SIMPLE', 'activation': -0.02},
                         {'id': 'input_5', 'is_active': True, 'type': 'SIMPLE', 'activation': -0.05}]

        }

        fcm = self.__init_complex_fcm()
        fcm.set_map_activation_function('biestate')
        fcm.run_inference()
        json_fcm = json.loads(fcm.to_json())
        self.assertEqual(expected_json["concepts"], json_fcm["concepts"])

    def test_inference_threestate(self):
        expected_json = {
            'concepts': [{'id': 'result_1', 'is_active': True, 'type': 'DECISION', 'activation': 0.0625},
                         {'id': 'result_2', 'is_active': True, 'type': 'DECISION', 'activation': 0.0},
                         {'id': 'input_1', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.0625},
                         {'id': 'input_2', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.025},
                         {'id': 'input_3', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.125},
                         {'id': 'input_4', 'is_active': True, 'type': 'SIMPLE', 'activation': -0.025},
                         {'id': 'input_5', 'is_active': True, 'type': 'SIMPLE', 'activation': -0.0625}]
        }

        fcm = self.__init_complex_fcm()
        fcm.set_map_activation_function('threestate')
        fcm.run_inference()
        json_fcm = json.loads(fcm.to_json())
        self.assertEqual(expected_json["concepts"], json_fcm["concepts"])

    def test_inference_saturation(self):
        expected_json = {
            'concepts': [{'id': 'result_1', 'is_active': True, 'type': 'DECISION', 'activation': 0.06},
                         {'id': 'result_2', 'is_active': True, 'type': 'DECISION', 'activation': 0.0},
                         {'id': 'input_1', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.05},
                         {'id': 'input_2', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.034999999999999996},
                         {'id': 'input_3', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.13},
                         {'id': 'input_4', 'is_active': True, 'type': 'SIMPLE', 'activation': -0.02},
                         {'id': 'input_5', 'is_active': True, 'type': 'SIMPLE', 'activation': -0.05}]
        }

        fcm = self.__init_complex_fcm()
        fcm.set_map_activation_function('saturation')
        fcm.run_inference()
        json_fcm = json.loads(fcm.to_json())
        self.assertEqual(expected_json["concepts"], json_fcm["concepts"])

    def test_inference_sum_w(self):
        expected_json = {
            'concepts': [{'id': 'result_1', 'is_active': True, 'type': 'DECISION', 'activation': 0.06},
                         {'id': 'result_2', 'is_active': True, 'type': 'DECISION', 'activation': 0.0},
                         {'id': 'input_1', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.05},
                         {'id': 'input_2', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.034999999999999996},
                         {'id': 'input_3', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.13},
                         {'id': 'input_4', 'is_active': True, 'type': 'SIMPLE', 'activation': -0.02},
                         {'id': 'input_5', 'is_active': True, 'type': 'SIMPLE', 'activation': -0.05}]
        }

        fcm = self.__init_complex_fcm()
        fcm.set_map_activation_function('sum_w', weight=0.0001)
        fcm.run_inference()
        json_fcm = json.loads(fcm.to_json())
        self.assertEqual(expected_json["concepts"], json_fcm["concepts"])

    def test_inference_several_functions(self):
        expected_json = {
            'concepts': [
                {'id': 'result_1', 'is_active': True, 'type': 'DECISION', 'activation': 0.0, 'use_memory': True,
                 'custom_function': 'sum_w', 'custom_function_args': {'weight': 0.003}},
                {'id': 'result_2', 'is_active': True, 'type': 'DECISION', 'activation': 0.00012648385157387873,
                 'use_memory': True, 'custom_function': 'sigmoid', 'custom_function_args': {'lambda_val': 10}},
                {'id': 'input_1', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.07375, 'use_memory': True,
                 'custom_function': 'threestate'},
                {'id': 'input_2', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.04361062920291475},
                {'id': 'input_3', 'is_active': True, 'type': 'SIMPLE', 'activation': -0.6816478125020062},
                {'id': 'input_4', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.18674101269496457},
                {'id': 'input_5', 'is_active': True, 'type': 'SIMPLE', 'activation': -0.0625}]
        }

        fcm = self.__init_complex_fcm()
        fcm.add_concept('result_1', concept_type=TYPE_DECISION, use_memory=True,
                        exitation_function='PAPAGEORGIUS', activation_function='sum_w', weight=0.003)

        fcm.add_concept('result_2', concept_type=TYPE_DECISION, use_memory=True,
                        exitation_function='PAPAGEORGIUS', activation_function='sigmoid', lambda_val=10)

        fcm.add_concept('input_1', use_memory=True, exitation_function='PAPAGEORGIUS',
                        activation_function='threestate')
        fcm.init_concept('input_1', 0.59)
        fcm.run_inference()
        json_fcm = json.loads(fcm.to_json())
        self.assertEqual(expected_json["concepts"], json_fcm["concepts"])
