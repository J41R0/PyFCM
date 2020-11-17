import json
import unittest
import pandas as pd
from py_fcm.learning.association import AssociationBasedFCM


class GeneratorTests(unittest.TestCase):
    @staticmethod
    def gen_fcm():
        generator = AssociationBasedFCM()

        test_input = [
            ['x', 5, 2.3, 'v1'],
            ['y', 7, 4.8, 'v1'],
            ['z', 3, 28.01, 'v2'],
            ['w', 1, 15.7, 'v2']
        ]

        my_ds = pd.DataFrame(test_input, columns=['f1', 'f2', 'f3', 'class'])
        generated_fcm = generator.gen_fcm(my_ds, target_features=['class'])
        return generated_fcm

    def test_association_generator_concepts(self):
        expected_json = {"concepts": [{"id": "w___f1", "is_active": True, "type": "SIMPLE", "activation": 0.0},
                                      {"id": "x___f1", "is_active": True, "type": "SIMPLE", "activation": 0.0},
                                      {"id": "y___f1", "is_active": True, "type": "SIMPLE", "activation": 0.0},
                                      {"id": "z___f1", "is_active": True, "type": "SIMPLE", "activation": 0.0},
                                      {"id": "0___f2", "is_active": True, "type": "SIMPLE", "activation": 0.0,
                                       "custom_function": "fuzzy", "activation_dict": {
                                          "membership": [0.14285714285714285, 0.42857142857142855, 0.7142857142857143,
                                                         1.0], "val_list": [1.0, 3.0, 5.0, 7.0]}},
                                      {"id": "0___f3", "is_active": True, "type": "SIMPLE", "activation": 0.0,
                                       "custom_function": "fuzzy", "activation_dict": {
                                          "membership": [0.0821135308818279, 0.17136736879685824, 0.5605141021063905,
                                                         1.0], "val_list": [2.3, 4.8, 15.7, 28.01]}},
                                      {"id": "v1___class", "is_active": True, "type": "DECISION", "activation": 0.0},
                                      {"id": "v2___class", "is_active": True, "type": "DECISION", "activation": 0.0}]
                         }

        generated_fcm = GeneratorTests.gen_fcm()
        json_fcm = json.loads(generated_fcm.to_json())
        self.assertEqual(expected_json['concepts'], json_fcm['concepts'])

    def test_association_generator_relations(self):
        expected_json = {"relations": [{"origin": "w___f1", "destiny": "x___f1", "weight": -1},
                                       {"origin": "x___f1", "destiny": "w___f1", "weight": -1},
                                       {"origin": "w___f1", "destiny": "y___f1", "weight": -1},
                                       {"origin": "y___f1", "destiny": "w___f1", "weight": -1},
                                       {"origin": "w___f1", "destiny": "z___f1", "weight": -1},
                                       {"origin": "z___f1", "destiny": "w___f1", "weight": -1},
                                       {"origin": "x___f1", "destiny": "y___f1", "weight": -1},
                                       {"origin": "y___f1", "destiny": "x___f1", "weight": -1},
                                       {"origin": "x___f1", "destiny": "z___f1", "weight": -1},
                                       {"origin": "z___f1", "destiny": "x___f1", "weight": -1},
                                       {"origin": "y___f1", "destiny": "z___f1", "weight": -1},
                                       {"origin": "z___f1", "destiny": "y___f1", "weight": -1},
                                       {"origin": "0___f2", "destiny": "w___f1", "weight": 0.4375},
                                       {"origin": "w___f1", "destiny": "0___f2", "weight": 1.0},
                                       {"origin": "0___f2", "destiny": "x___f1", "weight": 0.0625},
                                       {"origin": "x___f1", "destiny": "0___f2", "weight": 0.14285714285714285},
                                       {"origin": "0___f2", "destiny": "y___f1", "weight": 0.1875},
                                       {"origin": "y___f1", "destiny": "0___f2", "weight": 0.42857142857142855},
                                       {"origin": "0___f2", "destiny": "z___f1", "weight": 0.3125},
                                       {"origin": "z___f1", "destiny": "0___f2", "weight": 0.7142857142857143},
                                       {"origin": "0___f3", "destiny": "w___f1", "weight": 0.5512694351505609},
                                       {"origin": "w___f1", "destiny": "0___f3", "weight": 1.0},
                                       {"origin": "0___f3", "destiny": "x___f1", "weight": 0.045266679787443406},
                                       {"origin": "x___f1", "destiny": "0___f3", "weight": 0.0821135308818279},
                                       {"origin": "0___f3", "destiny": "y___f1", "weight": 0.09446959259988191},
                                       {"origin": "y___f1", "destiny": "0___f3", "weight": 0.17136736879685824},
                                       {"origin": "0___f3", "destiny": "z___f1", "weight": 0.30899429246211374},
                                       {"origin": "z___f1", "destiny": "0___f3", "weight": 0.5605141021063905},
                                       {"origin": "0___f3", "destiny": "0___f2", "weight": 0.8189332808502263},
                                       {"origin": "0___f2", "destiny": "0___f3", "weight": 0.6499241342377723},
                                       {"origin": "v1___class", "destiny": "v2___class", "weight": -1},
                                       {"origin": "v2___class", "destiny": "v1___class", "weight": -1},
                                       {"origin": "v1___class", "destiny": "0___f3", "weight": 0.12674044983934307},
                                       {"origin": "0___f3", "destiny": "v1___class", "weight": 0.13973627238732533},
                                       {"origin": "v2___class", "destiny": "0___f3", "weight": 0.7802570510531952},
                                       {"origin": "0___f3", "destiny": "v2___class", "weight": 0.8602637276126747},
                                       {"origin": "v1___class", "destiny": "x___f1", "weight": 0.5},
                                       {"origin": "x___f1", "destiny": "v1___class", "weight": 1.0},
                                       {"origin": "v1___class", "destiny": "y___f1", "weight": 0.5},
                                       {"origin": "y___f1", "destiny": "v1___class", "weight": 1.0},
                                       {"origin": "v2___class", "destiny": "w___f1", "weight": 0.5},
                                       {"origin": "w___f1", "destiny": "v2___class", "weight": 1.0},
                                       {"origin": "v2___class", "destiny": "z___f1", "weight": 0.5},
                                       {"origin": "z___f1", "destiny": "v2___class", "weight": 1.0},
                                       {"origin": "v1___class", "destiny": "0___f2", "weight": 0.2857142857142857},
                                       {"origin": "0___f2", "destiny": "v1___class", "weight": 0.25},
                                       {"origin": "v2___class", "destiny": "0___f2", "weight": 0.8571428571428572},
                                       {"origin": "0___f2", "destiny": "v2___class", "weight": 0.7500000000000001}]
                         }

        generated_fcm = GeneratorTests.gen_fcm()
        json_fcm = json.loads(generated_fcm.to_json())
        for elment_pos in range(len(expected_json['relations'])):
            self.assertIn(expected_json['relations'][elment_pos], json_fcm['relations'])
