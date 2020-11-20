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
    def __init__(self, str_separator="___", fit_inclination=False, exclusion_val=-1, causality_function=Relation.conf,
                 causal_eval_function=Relation.conf, causal_threshold=0, sign_function=None, sign_threshold=0):
        self.__features = {}
        self.__processed_features = set()

        # vars
        self.__fcm = FuzzyCognitiveMap()
        self.__processed_features = set()
        self.__exclusion_val = exclusion_val
        self.__causality_value_function = causality_function
        self.__causality_evaluation_function = causal_eval_function
        self.__causal_threshold = causal_threshold
        self.__sign_function = sign_function
        self.__sign_function_cut_val = sign_threshold
        self.__str_separator = str_separator

        if fit_inclination:
            self.__fcm.fit_inclination = 0.975

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
        for curr_cluster in range(n_clusters):
            concept_name = self.__name_concept(feat_name, curr_cluster)
            names.append(concept_name)
            fun_args = {'membership': memberships[curr_cluster],
                        'val_list': val_list}
            self.__fcm.add_concept(concept_name, self.__features[feat_name][TYPE], is_active=True,
                                   activation_dict=fun_args)

        self.__features[feat_name][CONCEPT_NAMES] = names
        self.__features[feat_name][CONCEPT_DESC] = ([i for i in range(n_clusters)], memberships)

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
        if self.__features[feat_name][TYPE] == TYPE_SIMPLE or self.__features[feat_name][TYPE] == TYPE_DECISION:
            related_concepts = self.__features[feat_name][CONCEPT_DESC][FEATURE_CONCEPTS]
            for concept1_pos in range(len(related_concepts) - 1):
                for concept2_pos in range(concept1_pos + 1, len(related_concepts)):
                    self.__fcm.add_relation(self.__features[feat_name][CONCEPT_NAMES][concept1_pos],
                                            self.__features[feat_name][CONCEPT_NAMES][concept2_pos],
                                            self.__exclusion_val)
                    self.__fcm.add_relation(self.__features[feat_name][CONCEPT_NAMES][concept2_pos],
                                            self.__features[feat_name][CONCEPT_NAMES][concept1_pos],
                                            self.__exclusion_val)

    def __def_feat_relations(self, feat_name):
        self.__def_inner_feat_relations(feat_name)
        for other_feat in self.__processed_features:
            if other_feat != feat_name:
                for concept_p_pos in range(len(self.__features[feat_name][CONCEPT_NAMES])):
                    for concept_q_pos in range(len(self.__features[other_feat][CONCEPT_NAMES])):
                        relation_coefficients = calc_concepts_coefficient(
                            self.__features[feat_name][CONCEPT_DESC][FEATURE_DESC][concept_p_pos],
                            self.__features[other_feat][CONCEPT_DESC][FEATURE_DESC][concept_q_pos]
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

                        # define causality degree p -> q
                        causality_p_q = self.__causality_evaluation_function(*relation_coefficients[P_Q_COEFF])
                        if causality_p_q > self.__causal_threshold:
                            relation_weight = sign_p_q * self.__causality_value_function(
                                *relation_coefficients[P_Q_COEFF])
                            self.__fcm.add_relation(self.__features[feat_name][CONCEPT_NAMES][concept_p_pos],
                                                    self.__features[other_feat][CONCEPT_NAMES][concept_q_pos],
                                                    relation_weight)

                        # define causality degree q -> p
                        causality_q_p = self.__causality_evaluation_function(*relation_coefficients[Q_P_COEFF])
                        if causality_q_p > self.__causal_threshold:
                            relation_weight = sign_q_p * self.__causality_value_function(
                                *relation_coefficients[Q_P_COEFF])
                            self.__fcm.add_relation(self.__features[other_feat][CONCEPT_NAMES][concept_q_pos],
                                                    self.__features[feat_name][CONCEPT_NAMES][concept_p_pos],
                                                    relation_weight)

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
            if dataset[feat_name].dtype == np.object:
                self.__gen_discrete_concepts(feat_name, target_features)

            # continuous feature
            elif dataset[feat_name].dtype == np.float64:
                self.__gen_continuous_concepts(feat_name, target_features, plot, plot_dir)

            else:
                uniques_data = np.unique(self.__features[feat_name][NP_ARRAY_DATA], return_counts=True)
                # frequency based node type inference
                if (uniques_data[UNIQUE_ARRAY].size / self.__features[feat_name][NP_ARRAY_DATA].size) <= 0.2:
                    self.__gen_discrete_concepts(feat_name, target_features, uniques_data)
                else:
                    # self.__features[feat_name][NP_ARRAY_DATA] = self.__features[feat_name][NP_ARRAY_DATA].astype(
                    #     np.float64)
                    self.__gen_continuous_concepts(feat_name, target_features, plot, plot_dir)

            self.__def_feat_relations(feat_name)

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
