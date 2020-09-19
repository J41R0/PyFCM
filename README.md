# py-fcm
Fuzzy cognitive maps python library

###Example usage

```
from py_fcm import from_json

fcm_json = """{
         "iter": 500,
         "activation_function": "sigmoid",
         "actv_func_params": {"lambda_val":1},
         "memory_influence": false,
         "result": "last",
         "concepts" :
          [
            {"id": "concept_1", "type": "SIMPLE", "activation": 0.5},
            {"id": "concept_2", "type": "DECISION", "custom_function": "sum_w", "custom_func_args": {"weight":0.3}},
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
        """
my_fcm = from_json(fcm_json)
my_fcm.run_inference()
result = my_fcm.get_result_by_type(node_type='any')
print(result)
```