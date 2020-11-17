from pandas import DataFrame

from py_fcm.fcm import FuzzyCognitiveMap
from py_fcm.utils.functions import Relation
from py_fcm.learning.association import AssociationBasedFCM


class FcmEstimator:
    def __init__(self, fcm_max_it=250, fcm_extra_steps=5, fcm_stability_diff=0.001, fcm_decision_function="MEAN",
                 fcm_excitation_function='KOSKO', fcm_mem_influence=False, concept_exclusion_val=-1,
                 concept_str_separator="___", fit_inclination=False, causal_eval_function=Relation.conf,
                 causal_threshold=0, causality_function=Relation.conf, causal_sign_function=None,
                 causal_sign_threshold=0, fcm_activation_function="sigmoid_hip", **kwargs):
        # init FCM
        self.__fcm = FuzzyCognitiveMap(max_it=fcm_max_it,
                                       extra_steps=fcm_extra_steps,
                                       stability_diff=fcm_stability_diff,
                                       decision_function=fcm_decision_function,
                                       mem_influence=fcm_mem_influence,
                                       activation_function=fcm_activation_function,
                                       **kwargs)
        self.__fcm.set_default_concept_properties(excitation_function=fcm_excitation_function)
        # init generator
        self.__generator = AssociationBasedFCM(str_separator=concept_str_separator,
                                               fit_inclination=fit_inclination,
                                               exclusion_val=concept_exclusion_val,
                                               causality_function=causality_function,
                                               causal_eval_function=causal_eval_function,
                                               causal_threshold=causal_threshold,
                                               sign_function=causal_sign_function,
                                               sign_threshold=causal_sign_threshold)

    def fit(self, x: DataFrame, y: DataFrame):
        target_feat = []
        for col in y.columns:
            target_feat.append(col)
        x.join(y, rsuffix='_class')
        self.__generator.build_fcm(x, target_features=target_feat, fcm=self.__fcm)

    def predict(self, x: DataFrame):
        pass

    def score(self, x: DataFrame, y: DataFrame):
        pass
