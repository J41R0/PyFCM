# PyFCM
Fuzzy cognitive maps python library. Also, supports the topology generation from data to solve classification problems.
The details associated to the generation process are described in [this paper](https://link.springer.com/chapter/10.1007/978-3-030-89691-1_25). 
### Installation

#### From source:

1. Clone repository:
    ```
    $ git clone https://github.com/J41R0/PyFCM.git 
    $ cd PyFCM
    ```
2. Install setup tools and package:
    ```
    $ pip install setuptools
    $ python setup.py install
    ```
#### From PyPi:
1. Install package using pip:
    ```
    $ pip install py-fcm
    ```
   
### Example usage

#### Inference:
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
                        "custom_function": "gceq",
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
result = my_fcm.get_final_state(concept_type='any')
print(result)
```

#### Generation:
```
import pandas
from py_fcm import FcmEstimator

data_dict = {
   'F1': ['x', 'x', 'y', 'y'],
   'F2': [9.8, 7.3, 1.1, 3.6],
   'class': ['a', 'a', 'r', 'r']
}
    
 train = pandas.DataFrame(data_dict)
 x_train = train.loc[:, train.columns != 'class']
 y_train = train.loc[:, 'class']

 estimator = FcmEstimator()
 estimator.fit(x_train, y_train)
 print(estimator.predict(x_train))
 print("Accuracy: ",estimator.score(x_train, y_train))
 print(estimator.get_fcm().to_json())

```