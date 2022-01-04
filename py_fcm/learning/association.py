import warnings
from collections import defaultdict

from pandas import DataFrame

from py_fcm.utils.__const import *
from py_fcm.learning.utils import *
from py_fcm.fcm import FuzzyCognitiveMap
from py_fcm.utils.functions import Relation
from py_fcm.learning.cmeans_discretization import fuzzy_feature_discretization

# USED CONST DEFINITION
# features
NP_ARRAY_DATA = "np array data"
CONCEPT_DESC = "feature concepts description"
CONCEPT_NAMES = "feature concept names"
TYPE = "concept type"
FEATURE_CONCEPTS = 0
FEATURE_DESC = 1

# np unique output consts
UNIQUE_ARRAY = 0

# coefficient calc output
P_Q_COEFF = 0
Q_P_COEFF = 1


class AssociationBasedFCM:
    def __init__(self, str_separator="___", discretization="cmeans-gap", exclusion_val=-1, fit_inclination=True,
                 double_relation=True, features_relation=True, causality_function=Relation.conf,
                 causal_eval_function=Relation.conf, causal_threshold=0.0, min_supp=0.0, sign_function=None,
                 sign_threshold=0):
        self.__features = {}
        self.__processed_features = set()

        # vars
        self.__fcm = FuzzyCognitiveMap()
        self.__exclusion_val = exclusion_val
        self.__causality_value_function = causality_function
        self.__causality_evaluation_function = causal_eval_function
        self.__causal_threshold = causal_threshold
        self.__min_support = min_supp
        self.__sign_function = sign_function
        self.__sign_function_cut_val = sign_threshold
        self.__str_separator = str_separator
        self.__double_relation = double_relation
        self.__features_relation = features_relation
        self.__discretization_method = discretization

        if fit_inclination:
            self.__fcm.fit_inclination = 0.75
        self.infer_concept_type = True

    def __name_concept(self, feat_name, value):
        return str(value) + str(self.__str_separator) + str(feat_name)

    def __gen_continuous_concepts(self, feat_name, target_feats, plot, plot_dir):
        if feat_name in target_feats:
            self.__features[feat_name][TYPE] = TYPE_REGRESOR
        else:
            self.__features[feat_name][TYPE] = TYPE_FUZZY

        n_clusters, val_list, memberships = fuzzy_feature_discretization(self.__features[feat_name][NP_ARRAY_DATA],
                                                                         att_name=feat_name, plot=plot,
                                                                         plot_dir=plot_dir)
        names = []
        final_membership = np.zeros((n_clusters, self.__features[feat_name][NP_ARRAY_DATA].size))
        for curr_cluster in range(n_clusters):
            concept_name = self.__name_concept(feat_name, curr_cluster)
            names.append(concept_name)
            fun_args = {'membership': memberships[curr_cluster],
                        'val_list': val_list}
            self.__fcm.add_concept(concept_name, self.__features[feat_name][TYPE], is_active=True,
                                   activation_dict=fun_args)
            for val_pos in range(self.__features[feat_name][NP_ARRAY_DATA].size):
                curr_value = self.__features[feat_name][NP_ARRAY_DATA][val_pos]
                final_membership[curr_cluster][val_pos] = fuzzy_set(value=curr_value,
                                                                    membership=fun_args['membership'],
                                                                    val_list=fun_args['val_list'])

        self.__features[feat_name][CONCEPT_NAMES] = names
        self.__features[feat_name][CONCEPT_DESC] = ([i for i in range(n_clusters)], final_membership)

    def __gen_discrete_concepts(self, feat_name, target_feats, uniques_data=None):
        if feat_name in target_feats:
            self.__features[feat_name][TYPE] = TYPE_DECISION
        else:
            self.__features[feat_name][TYPE] = TYPE_SIMPLE

        if uniques_data is None:
            uniques_data = np.unique(self.__features[feat_name][NP_ARRAY_DATA], return_counts=True)

        feat_matrix = gen_discrete_feature_matrix(self.__features[feat_name][NP_ARRAY_DATA], uniques_data[UNIQUE_ARRAY])

        names = []
        for val_pos in range(uniques_data[UNIQUE_ARRAY].size):
            name = self.__name_concept(feat_name, uniques_data[UNIQUE_ARRAY][val_pos])
            names.append(name)
            self.__fcm.add_concept(name, self.__features[feat_name][TYPE], is_active=True)
        self.__features[feat_name][CONCEPT_NAMES] = names
        self.__features[feat_name][CONCEPT_DESC] = (uniques_data[UNIQUE_ARRAY], feat_matrix)

    def __def_inner_feat_relations(self, feat_name):
        # FIXME: This function only works for fuzzy and discrete features, use the function type instead.
        if self.__exclusion_val != 0:
            related_concepts = self.__features[feat_name][CONCEPT_DESC][FEATURE_CONCEPTS]
            for concept1_pos in range(len(related_concepts) - 1):
                for concept2_pos in range(concept1_pos + 1, len(related_concepts)):
                    self.__fcm.add_relation(self.__features[feat_name][CONCEPT_NAMES][concept1_pos],
                                            self.__features[feat_name][CONCEPT_NAMES][concept2_pos],
                                            self.__exclusion_val)
                    self.__fcm.add_relation(self.__features[feat_name][CONCEPT_NAMES][concept2_pos],
                                            self.__features[feat_name][CONCEPT_NAMES][concept1_pos],
                                            self.__exclusion_val)

    def __def_relation_weight(self, feat_name_a, concept_a_pos, feat_name_b, concept_b_pos, sign, p_q, p_nq, np_q,
                              np_nq):
        causality_p_q = self.__causality_evaluation_function(p_q, p_nq, np_q, np_nq)
        if abs(causality_p_q) > self.__causal_threshold and abs(causality_p_q) > self.__min_support:
            relation_weight = sign * self.__causality_value_function(p_q, p_nq, np_q, np_nq)
            if relation_weight != 0:
                self.__fcm.add_relation(self.__features[feat_name_a][CONCEPT_NAMES][concept_a_pos],
                                        self.__features[feat_name_b][CONCEPT_NAMES][concept_b_pos],
                                        relation_weight)

    def __def_two_feat_relation(self, feat_name_a, feat_name_b):
        for concept_p_pos in range(len(self.__features[feat_name_a][CONCEPT_NAMES])):
            for concept_q_pos in range(len(self.__features[feat_name_b][CONCEPT_NAMES])):
                relation_coefficients = calc_concepts_coefficient(
                    self.__features[feat_name_a][CONCEPT_DESC][FEATURE_DESC][concept_p_pos],
                    self.__features[feat_name_b][CONCEPT_DESC][FEATURE_DESC][concept_q_pos]
                )
                if relation_coefficients is None:
                    raise Exception("Invalid relation input data")
                # define relation sign
                sign_p_q = 1
                sign_q_p = 1
                if self.__sign_function is not None:
                    if self.__sign_function_cut_val > self.__sign_function(*relation_coefficients[P_Q_COEFF]):
                        sign_p_q = -1
                    if self.__sign_function_cut_val > self.__sign_function(*relation_coefficients[Q_P_COEFF]):
                        sign_q_p = -1
                if self.__double_relation:
                    # define causality degree p -> q
                    self.__def_relation_weight(feat_name_a, concept_p_pos, feat_name_b, concept_q_pos, sign_p_q,
                                               *relation_coefficients[P_Q_COEFF])
                    # define causality degree q -> p
                    self.__def_relation_weight(feat_name_b, concept_q_pos, feat_name_a, concept_p_pos, sign_q_p,
                                               *relation_coefficients[Q_P_COEFF])
                else:
                    if self.__are_same_feature_group(feat_name_a, feat_name_b):
                        p_q = abs(self.__causality_value_function(*relation_coefficients[P_Q_COEFF]))
                        q_p = abs(self.__causality_value_function(*relation_coefficients[Q_P_COEFF]))
                        if p_q > q_p:
                            # define causality degree p -> q
                            self.__def_relation_weight(feat_name_a, concept_p_pos, feat_name_b, concept_q_pos,
                                                       sign_p_q, *relation_coefficients[P_Q_COEFF])
                        if q_p > p_q:
                            # define causality degree q -> p
                            self.__def_relation_weight(feat_name_b, concept_q_pos, feat_name_a, concept_p_pos,
                                                       sign_q_p, *relation_coefficients[Q_P_COEFF])
                    else:
                        if not self.__is_target_concept(feat_name_a):
                            self.__def_relation_weight(feat_name_a, concept_p_pos, feat_name_b, concept_q_pos,
                                                       sign_p_q, *relation_coefficients[P_Q_COEFF])
                        else:
                            self.__def_relation_weight(feat_name_b, concept_q_pos, feat_name_a, concept_p_pos,
                                                       sign_q_p, *relation_coefficients[Q_P_COEFF])

    def __def_all_feat_relations(self, feat_name):
        for other_feat in self.__processed_features:
            if other_feat != feat_name:
                self.__def_two_feat_relation(feat_name, other_feat)

        self.__processed_features.add(feat_name)

    def __is_target_concept(self, feat_name):
        if self.__features[feat_name][TYPE] == TYPE_REGRESOR or self.__features[feat_name][TYPE] == TYPE_DECISION:
            return True
        return False

    def __are_same_feature_group(self, name_feat1, name_feat2):
        return ((self.__is_target_concept(name_feat1) and self.__is_target_concept(name_feat2)) or
                not self.__is_target_concept(name_feat1) and not self.__is_target_concept(name_feat2))

    def __def_feat_target_relation(self, feat_name):
        for other_feat in self.__processed_features:
            if other_feat != feat_name and not self.__are_same_feature_group(feat_name, other_feat):
                self.__def_two_feat_relation(feat_name, other_feat)
        self.__processed_features.add(feat_name)

    def build_fcm(self, dataset: DataFrame, target_features=None, fcm=None, plot=False,
                  plot_dir='.') -> FuzzyCognitiveMap:
        # TODO: handle features multivalued and with missing values
        if fcm is not None and type(fcm) == FuzzyCognitiveMap:
            self.__fcm = fcm
        else:
            self.__fcm = FuzzyCognitiveMap()
        if target_features is None:
            target_features = set()
        # define map concepts
        for feat_name in dataset.loc[:, ]:
            self.__features[feat_name] = {NP_ARRAY_DATA: np.array(dataset.loc[:, feat_name].values)}
            # discrete features
            # TODO: extend to categorical series
            if dataset[feat_name].dtype == np.object or dataset[feat_name].dtype == np.bool:
                self.__gen_discrete_concepts(feat_name, target_features)

            # possible continuous feature
            elif self.infer_concept_type:
                uniques_data = np.unique(self.__features[feat_name][NP_ARRAY_DATA], return_counts=True)
                if uniques_data[UNIQUE_ARRAY].size < 10:
                    self.__gen_discrete_concepts(feat_name, target_features, uniques_data)
                else:
                    # TODO: review type inference
                    if (uniques_data[UNIQUE_ARRAY].size / self.__features[feat_name][NP_ARRAY_DATA].size) > 0.2:
                        warnings.warn("Possible discrete feature behavior for " + feat_name + " feature.")
                    self.__features[feat_name][NP_ARRAY_DATA] = self.__features[feat_name][
                        NP_ARRAY_DATA].astype(
                        np.float64)
                    self.__gen_continuous_concepts(feat_name, target_features, plot, plot_dir)
            else:
                self.__gen_continuous_concepts(feat_name, target_features, plot, plot_dir)

            # TODO: identify feature type to define the inner concept's relation properly
            self.__def_inner_feat_relations(feat_name)
            self.__processed_features.add(feat_name)

            if self.__features_relation:
                self.__def_all_feat_relations(feat_name)
            else:
                self.__def_feat_target_relation(feat_name)

        return self.__fcm

    def init_concept_by_feature_data(self, feat_name, value):
        try:
            if self.__features[feat_name][TYPE] == TYPE_FUZZY or self.__features[feat_name][TYPE] == TYPE_REGRESOR:
                for concept in self.__features[feat_name][CONCEPT_NAMES]:
                    self.__fcm.init_concept(concept, value, required_presence=False)
            else:
                if type(value) != str:
                    value = int(value)
                self.__fcm.init_concept(self.__name_concept(feat_name, value), 1.0, required_presence=False)
        except Exception:
            raise Exception("Cannot init concept")

    def __get_feature_and_info(self, concept: str):
        res = concept.split(self.__str_separator)
        return res[1], res[0]

    def __get_discrete_feature_result(self, fcm_results):
        final_result = {}
        for feat_name in fcm_results:
            max_actv = -1
            res_pos = 0
            curr_pos = 0
            for value, output in fcm_results[feat_name]:
                if output > max_actv:
                    max_actv = output
                    res_pos = curr_pos
                curr_pos += 1
            # return result with highest activation
            result = fcm_results[feat_name][res_pos][0]
            if result.isnumeric():
                result = int(result)
            final_result[feat_name] = result
        return final_result

    def get_inference_result(self, plot=False, map_name="fcm", plot_dir='.'):
        self.__fcm.run_inference()
        fcm_result = self.__fcm.get_final_state()
        cont_res_feat = defaultdict(list)
        disc_res_feat = defaultdict(list)
        for concept in fcm_result:
            feat_name, info = self.__get_feature_and_info(concept)
            if self.__features[feat_name][TYPE] == TYPE_FUZZY or self.__features[feat_name][TYPE] == TYPE_REGRESOR:
                cont_res_feat[feat_name].append((info, fcm_result[concept]))
            else:
                disc_res_feat[feat_name].append((info, fcm_result[concept]))
        if plot:
            self.__fcm.plot_execution(fig_name=map_name, plot_dir=plot_dir)
        # TODO: handle continuous features output for regression problems
        return self.__get_discrete_feature_result(disc_res_feat)

    def reset(self):
        self.__features = {}
        self.__processed_features = set()
        self.__fcm.clear_all()
