# py-fcm
Fuzzy cognitive maps python library

###Example usage

```
from py_fcm import from_json

fcm_json = """{
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
        """
my_fcm = from_json(fcm_json)
my_fcm.run_inference()
result = my_fcm.get_result_by_type(node_type='any')
print(result)
```