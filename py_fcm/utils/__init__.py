import ast
from collections import OrderedDict

import arff
import pandas


def load_dataset(ds_path, factorize=False, max_int_uniques=10, int_factor=0.2):
    ATT_NAME = 0
    data_dict = OrderedDict()
    ds_name = ds_path.split('/')[-1]
    ds_ext = ds_name.split('.')[-1]
    known_format = False
    last_feat = ''
    try:
        if 'arff' == ds_ext:
            with open(ds_path) as file:
                text = file.read()
                arff_dict = arff.loads(text)
                for attribute in arff_dict['attributes']:
                    data_dict[attribute[ATT_NAME]] = []
                for row in arff_dict['data']:
                    for att_pos in range(0, len(arff_dict['attributes'])):
                        try:
                            data_dict[arff_dict['attributes'][att_pos][ATT_NAME]].append(ast.literal_eval(row[att_pos]))
                            last_feat = arff_dict['attributes'][att_pos][ATT_NAME]
                        except:
                            data_dict[arff_dict['attributes'][att_pos][ATT_NAME]].append(row[att_pos])
            known_format = True
        if 'csv' == ds_ext:
            with open(ds_path) as file:
                content = []
                for line in file:
                    content.append(line.strip())
            # first line describe the atributes names
            attributes = str(content[0]).split(',')
            for current_att in attributes:
                data_dict[current_att] = []
            for line in range(1, len(content)):
                data_line = str(content[line]).split(',')
                # avoid rows with different attributes length
                if len(data_line) == len(attributes):
                    # the missing data must be identified
                    for data in range(0, len(data_line)):
                        # reusing for value type inference
                        try:
                            data_dict[attributes[data]].append(ast.literal_eval(data_line[data]))
                            last_feat = attributes[data]
                        except:
                            data_dict[attributes[data]].append(data_line[data])
                else:
                    # Handle errors in dataset matrix
                    raise Exception("Errors in line: ", line, len(data_line), len(attributes))
            known_format = True
    except Exception as err:
        print("Error: ", str(err))
    if not known_format:
        raise Exception("Unknown dataset format")
    else:
        if 'class' in data_dict:
            codes, uniques = pandas.factorize(data_dict['class'])
            classes = len(uniques)
            objects = len(codes)
        else:
            codes, uniques = pandas.factorize(data_dict[last_feat])
            classes = len(uniques)
            objects = len(codes)
        print("\n===> Dataset for test: ", ds_name)
        print('Features: ', len(data_dict), " | Objects: ", objects, " | Classes: ", classes)

        try:
            if factorize:
                for key in data_dict:
                    if type(data_dict[key][0]) == str:
                        codes, uniques = pandas.factorize(data_dict[key])
                        data_dict[key] = codes
                    elif type(data_dict[key][0]) == int:
                        codes, uniques = pandas.factorize(data_dict[key])
                        if len(uniques) <= max_int_uniques or (len(uniques) / len(codes)) < int_factor:
                            data_dict[key] = codes
            # remove equal entries
            dataset_frame = pandas.DataFrame(data_dict).drop_duplicates()
        except Exception as err:
            for key, value in data_dict.items():
                print(key, value)
                pass
            raise Exception(ds_name + " " + str(err))

        return dataset_frame
