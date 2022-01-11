from py_fcm.utils.functions import Relation
from py_fcm.fcm_estimator import FcmEstimator
from py_fcm.loader import from_json, join_maps, FuzzyCognitiveMap
from py_fcm.utils.__const import TYPE_SIMPLE, TYPE_DECISION, TYPE_FUZZY


def load_csv_dataset(dataset_dir, ds_name, factorize=False):
    import pandas
    from collections import OrderedDict
    data_dict = OrderedDict()
    with open(dataset_dir + "/" + ds_name) as file:
        content = []
        for line in file:
            content.append(line.strip())
    # first line describe the atributes names
    attributes = str(content[0]).split(',')
    for current_att in attributes:
        data_dict[current_att] = []

    # print(data_dict)
    for line in range(1, len(content)):
        data_line = str(content[line]).split(',')
        # avoid rows with different attributes length
        if len(data_line) == len(attributes):
            # the missing data must be identified
            for data in range(0, len(data_line)):
                # reusing for value type inference
                current_att = data_line[data]
                data_dict[attributes[data]].append(current_att)
        else:
            # Handle errors in dataset matrix
            print("Errors in line: ", line, len(data_line), len(attributes))
    # adding data set
    print("\n===> Dataset for test: ", ds_name)
    try:
        dataset_frame = pandas.DataFrame(data_dict).drop_duplicates()
    except Exception as err:
        for key, value in data_dict.items():
            print(key, value)
            pass
        raise Exception(ds_name + " " + str(err))
    # Transform the columns disc values in 0..N values
    if factorize:
        # for current_col in dataset_frame.
        dataset_frame = dataset_frame.apply(lambda x: pandas.factorize(x)[0])
    return dataset_frame
