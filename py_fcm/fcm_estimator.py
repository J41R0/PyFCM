from collections import defaultdict

import numpy as np
from pandas import DataFrame, Series

from py_fcm.utils.functions import Relation
from py_fcm.learning.association import AssociationBasedFCM, FuzzyCognitiveMap


class FcmEstimator:
    def __init__(self, concept_str_separator="___", concept_exclusion_val=-1, fit_inclination=True,
                 double_relation=True, features_relation=True, infer_concept_type=True, discretization="cmeans-cfe",
                 causal_eval_function=Relation.supp, causal_threshold=0.0, causality_function=Relation.pos_inf,
                 causal_sign_function=None, causal_sign_threshold=0, fcm_mem_influence=True, fcm_max_it=250,
                 fcm_extra_steps=5, fcm_stability_diff=0.001, fcm_decision_function="MEAN",
                 fcm_excitation_function='KOSKO', fcm_activation_function="sigmoid_hip", vectorized_run=True, **kwargs):
        self.__is_fcm_generated = False
        # init FCM
        self.__fcm = FuzzyCognitiveMap(max_it=fcm_max_it,
                                       extra_steps=fcm_extra_steps,
                                       stability_diff=fcm_stability_diff,
                                       decision_function=fcm_decision_function,
                                       mem_influence=fcm_mem_influence,
                                       activation_function=fcm_activation_function,
                                       **kwargs)
        self.__fcm.set_default_concept_properties(excitation_function=fcm_excitation_function)
        self.__fcm.debug = not vectorized_run
        # init generator
        self.__generator = AssociationBasedFCM(str_separator=concept_str_separator,
                                               discretization=discretization,
                                               fit_inclination=fit_inclination,
                                               double_relation=double_relation,
                                               features_relation=features_relation,
                                               exclusion_val=concept_exclusion_val,
                                               causality_function=causality_function,
                                               causal_eval_function=causal_eval_function,
                                               causal_threshold=causal_threshold,
                                               sign_function=causal_sign_function,
                                               sign_threshold=causal_sign_threshold)
        self.__generator.infer_concept_type = infer_concept_type

    def __to_data_frame(self, x, start=0):
        if type(x) == dict:
            return DataFrame(x)
        if type(x) == list:
            x = np.array(x)
        if type(x) == np.ndarray:
            col_names = []
            if len(x.shape) == 1:
                col_names.append(str(start))
            else:
                col_names = [str(i + start) for i in range(x.shape[1])]
            return DataFrame(x, columns=col_names)
        if type(x) == Series:
            return x.to_frame()
        return x

    def __validate_dual_input(self, x, y):
        if type(x) == list:
            x = np.array(x)
        max_col = x.shape[1]
        x = self.__to_data_frame(x)
        y = self.__to_data_frame(y, start=max_col)
        if type(x) != DataFrame:
            raise Exception("Cannot load provided x structure, please use a pandas DataFrame like object.")
        if type(y) != DataFrame:
            raise Exception("Cannot load provided y structure, please use a pandas DataFrame like object.")
        return x, y

    def fit(self, x, y, plot=False, plot_dir='.'):
        # expect a dataframe or a matrix shaped in a list of rows
        x, y = self.__validate_dual_input(x, y)
        target_feat = []
        for col in y.columns:
            target_feat.append(col)
        input_data = x.join(y, rsuffix='_class')
        self.__generator.reset()
        self.__generator.build_fcm(input_data, target_features=target_feat, fcm=self.__fcm,
                                   plot=plot, plot_dir=plot_dir)
        self.__is_fcm_generated = True

    def predict(self, x: DataFrame, plot=False, plot_dir='.'):
        if type(x) != DataFrame:
            x = self.__to_data_frame(x)
        if type(x) != DataFrame:
            raise Exception("Cannot load provided x structure, please use a pandas DataFrame like object.")
        result = defaultdict(list)
        if self.__is_fcm_generated:
            for index, row in x.iterrows():
                self.__fcm.clear_execution()
                for feat_name in x.columns:
                    self.__generator.init_concept_by_feature_data(feat_name, row[feat_name])

                prediction = self.__generator.get_inference_result(plot=plot, plot_dir=plot_dir)
                for feat_name in prediction:
                    result[feat_name].append(prediction[feat_name])
        else:
            raise Exception("There is no FCM generated, the fit method mus be called first")
        return DataFrame(result)

    def score(self, x, y):
        # TODO: Add other scoring functions besides accuracy
        if self.__is_fcm_generated:
            x, y = self.__validate_dual_input(x, y)
            predicted_result = self.predict(x)
            right_predicted = 0
            total = 0
            for feat_name in y.columns:
                pos = 0
                for val in y[feat_name]:
                    total += 1
                    if val == predicted_result[feat_name][pos]:
                        right_predicted += 1
                    pos += 1
            return right_predicted / total
        else:
            raise Exception("There is no FCM generated, the fit method mus be called first")

    def get_fcm(self):
        return self.__fcm
