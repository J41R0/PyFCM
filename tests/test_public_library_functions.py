import unittest
import json

from py_fcm import join_maps, from_json


class FromJsonTests(unittest.TestCase):
    def setUp(self) -> None:
        default_fcm = """
        {
         "max_iter": 500,
         "activation_function": "sigmoid",
         "activation_function_args": {"lambda_val":1},
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
        }"""
        self.fcm = from_json(default_fcm)

    def test_default(self):
        expected = {
            "max_iter": 500,
            "decision_function": "LAST",
            "activation_function": "sigmoid",
            "memory_influence": False,
            "stability_diff": 0.001,
            "stop_at_stabilize": True,
            "extra_steps": 5,
            "weight": 1,
            "concepts":
                [
                    {
                        "id": "concept_1",
                        "is_active": True,
                        "type": "SIMPLE",
                        "activation": 0.5
                    },
                    {
                        "id": "concept_2", "is_active": True,
                        "type": "DECISION", "activation": 0.0,
                        "custom_function": "sum_w",
                        "custom_function_args": {"weight": 0.3}
                    },
                    {
                        "id": "concept_3",
                        "is_active": True,
                        "type": "SIMPLE",
                        "activation": 0.0,
                        "use_memory": True
                    },
                    {
                        "id": "concept_4",
                        "is_active": True,
                        "type": "SIMPLE",
                        "activation": 0.3,
                        "custom_function": "saturation"
                    }
                ],
            "relations":
                [
                    {"origin": "concept_4", "destiny": "concept_2", "weight": -0.1},
                    {"origin": "concept_1", "destiny": "concept_3", "weight": 0.59},
                    {"origin": "concept_3", "destiny": "concept_2", "weight": 0.8911}
                ],
            'activation_function_args': {'lambda_val': 1},
        }
        fcm_json = json.loads(self.fcm.to_json())
        self.assertEqual(expected, fcm_json)


class JoinMapsTests(unittest.TestCase):
    def setUp(self) -> None:
        fcm_json1 = """
        {
             "max_iter": 500,
             "activation_function": "sigmoid",
             "actv_func_args": {"lambda_val":1},
             "memory_influence": false,
             "decision_function": "LAST",
             "concepts" :
              [
                {"id": "concept_1", "type": "SIMPLE", "activation": 0.25},
                {"id": "concept_2", "type": "DECISION", "custom_function": "sum_w", "custom_function_args": {"weight":0.3}}
              ],
             "relations":
              [
                {"origin": "concept_1", "destiny": "concept_2", "weight": 0.25}
              ]
        }"""

        fcm_json2 = """
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
                {"id": "concept_4", "type": "SIMPLE", "custom_function": "saturation", "activation": 0.3}
              ],
             "relations":
              [
                {"origin": "concept_4", "destiny": "concept_2", "weight": 0.2},
                {"origin": "concept_1", "destiny": "concept_2", "weight": 0.75}
              ]
        }"""

        fcm_json3 = """
        {
             "max_iter": 500,
             "activation_function": "sigmoid",
             "actv_func_args": {"lambda_val":1},
             "memory_influence": false,
             "decision_function": "LAST",
             "concepts" :
              [
                {"id": "concept_1", "type": "SIMPLE", "activation": 0.75},
                {"id": "concept_2", "type": "DECISION", "custom_function": "sum_w", "custom_function_args": {"weight":0.3}},
                {"id": "concept_3", "type": "SIMPLE", "memory_influence":true }
              ],
             "relations":
              [
                {"origin": "concept_1", "destiny": "concept_4", "weight": -0.3911},
                {"origin": "concept_2", "destiny": "concept_3", "weight": 0.8911}
              ]
        }"""
        self.fcm1 = from_json(fcm_json1)
        self.fcm2 = from_json(fcm_json2)
        self.fcm3 = from_json(fcm_json3)

    def test_default(self):
        fcm = join_maps([self.fcm1, self.fcm2, self.fcm3])
        expected = {
            "max_iter": 500,
            "decision_function": "LAST",
            "activation_function": "sigmoid",
            "memory_influence": False,
            "stability_diff": 0.001,
            "stop_at_stabilize": True,
            "extra_steps": 5,
            "weight": 1,
            "concepts": [
                {"id": "concept_1", "is_active": True, "type": "SIMPLE", "activation": 0.5},
                {"id": "concept_2", "is_active": True, "type": "DECISION", 'activation': 0.0,
                 "custom_function": "sum_w", "custom_function_args": {"weight": 0.3}},
                {"id": "concept_4", "is_active": True, "type": "SIMPLE", "activation": 0.3,
                 "custom_function": "saturation"},
                {"id": "concept_3", "is_active": True, "type": "SIMPLE", "activation": 0.0},
            ],
            "relations": [
                {'origin': 'concept_1', 'destiny': 'concept_2', 'weight': 0.5},
                {'origin': 'concept_4', 'destiny': 'concept_2', 'weight': 0.2},
                {'origin': 'concept_2', 'destiny': 'concept_3', 'weight': 0.8911}
            ]
        }

        fcm_json = json.loads(fcm.to_json())
        self.assertEqual(expected, fcm_json)

    def test_intersection(self):
        fcm = join_maps([self.fcm1, self.fcm2, self.fcm3], concept_strategy='intersection')
        expected = {
            "concepts": [
                {'id': 'concept_1', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.5},
                {'id': 'concept_2', 'is_active': True, 'type': 'DECISION', 'activation': 0.0,
                 'custom_function': 'sum_w', 'custom_function_args': {'weight': 0.3}}
            ],
            "relations": [
                {'origin': 'concept_1', 'destiny': 'concept_2', 'weight': 0.5}
            ]
        }

        fcm_json = json.loads(fcm.to_json())
        self.assertEqual(expected['concepts'], fcm_json['concepts'])
        self.assertEqual(expected['relations'], fcm_json['relations'])

    def test_highest_strategies(self):
        fcm = join_maps([self.fcm1, self.fcm2, self.fcm3],
                        concept_strategy='intersection',
                        value_strategy='highest',
                        relation_strategy='highest')
        expected = {
            "concepts": [
                {'id': 'concept_1', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.75},
                {'id': 'concept_2', 'is_active': True, 'type': 'DECISION', 'activation': 0.0,
                 'custom_function': 'sum_w', 'custom_function_args': {'weight': 0.3}}
            ],
            "relations": [
                {'origin': 'concept_1', 'destiny': 'concept_2', 'weight': 0.75}
            ]
        }

        fcm_json = json.loads(fcm.to_json())
        self.assertEqual(expected['concepts'], fcm_json['concepts'])
        self.assertEqual(expected['relations'], fcm_json['relations'])

    def test_lowest_strategies(self):
        fcm = join_maps([self.fcm1, self.fcm2, self.fcm3],
                        concept_strategy='intersection',
                        value_strategy='lowest',
                        relation_strategy='lowest')
        expected = {
            "concepts": [
                {'id': 'concept_1', 'is_active': True, 'type': 'SIMPLE', 'activation': 0.25},
                {'id': 'concept_2', 'is_active': True, 'type': 'DECISION', 'activation': 0.0,
                 'custom_function': 'sum_w', 'custom_function_args': {'weight': 0.3}}
            ],
            "relations": [
                {'origin': 'concept_1', 'destiny': 'concept_2', 'weight': 0.25}
            ]
        }

        fcm_json = json.loads(fcm.to_json())
        self.assertEqual(expected['concepts'], fcm_json['concepts'])
        self.assertEqual(expected['relations'], fcm_json['relations'])

    def test_exeptions(self):
        res = ''
        expected = 'Unknown concept strategy: aaaa'
        try:
            fcm = join_maps([self.fcm1, self.fcm2, self.fcm3], concept_strategy='aaaa')
        except Exception as err:
            res = str(err)
        self.assertEqual(expected, res)

        expected = 'Unknown value strategy: aaaa'
        try:
            fcm = join_maps([self.fcm1, self.fcm2, self.fcm3], value_strategy='aaaa')
        except Exception as err:
            res = str(err)
        self.assertEqual(expected, res)

        expected = 'Unknown relation strategy: aaaa'
        try:
            fcm = join_maps([self.fcm1, self.fcm2, self.fcm3], relation_strategy='aaaa')
        except Exception as err:
            res = str(err)
        self.assertEqual(expected, res)
