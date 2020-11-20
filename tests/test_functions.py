import unittest

from tests import create_concept
from py_fcm.utils.functions import *


class ExitationFunctionsTests(unittest.TestCase):
    def test_kosko(self):
        test_concept = create_concept("test", exitation_function='KOSKO')
        res = test_concept[NODE_EXEC_FUNC](test_concept)
        self.assertEqual(4, res)
        test_concept = create_concept("test", exitation_function='KOSKO', use_memory=False)
        res = test_concept[NODE_EXEC_FUNC](test_concept)
        self.assertEqual(3, res)

    def test_papageorgius(self):
        test_concept = create_concept("test", exitation_function='PAPAGEORGIUS')
        res = test_concept[NODE_EXEC_FUNC](test_concept)
        self.assertEqual(4, res)
        test_concept = create_concept("test", exitation_function='PAPAGEORGIUS', use_memory=False)
        res = test_concept[NODE_EXEC_FUNC](test_concept)
        self.assertEqual(3, res)


class ActivationFunctionsTests(unittest.TestCase):
    def test_biestate(self):
        test_concept = create_concept(activ_function='biestate')
        res = test_concept[NODE_ACTV_FUNC](0.5)
        self.assertEqual(1.0, res)
        res = test_concept[NODE_ACTV_FUNC](-0.5)
        self.assertEqual(0.0, res)

    def test_threestate(self):
        test_concept = create_concept(activ_function='threestate')
        res = test_concept[NODE_ACTV_FUNC](0.30)
        self.assertEqual(0.0, res)
        res = test_concept[NODE_ACTV_FUNC](0.60)
        self.assertEqual(0.5, res)
        res = test_concept[NODE_ACTV_FUNC](0.75)
        self.assertEqual(1.0, res)

    def test_saturation(self):
        test_concept = create_concept(activ_function='saturation')
        res = test_concept[NODE_ACTV_FUNC](-60)
        self.assertEqual(0.0, res)
        res = test_concept[NODE_ACTV_FUNC](75)
        self.assertEqual(1.0, res)

    def test_sigmoid(self):
        test_concept = create_concept(activ_function='sigmoid')
        res = test_concept[NODE_ACTV_FUNC](1)
        self.assertEqual(0.7310585786300049, res)
        res = test_concept[NODE_ACTV_FUNC](-1)
        self.assertEqual(0.2689414213699951, res)
        res = test_concept[NODE_ACTV_FUNC](-1, 50)
        self.assertEqual(1.928749847963918e-22, res)

    def test_sigmoid_and_tan_hip(self):
        test_concept = create_concept(activ_function='sigmoid_hip')
        res = test_concept[NODE_ACTV_FUNC](1)
        self.assertEqual(0.7615941559557649, res)
        res = test_concept[NODE_ACTV_FUNC](-1)
        self.assertEqual(-0.7615941559557649, res)
        test_concept = create_concept("test", activ_function='tan_hip')
        res = test_concept[NODE_ACTV_FUNC](1)
        self.assertEqual(0.7615941559557649, res)
        res = test_concept[NODE_ACTV_FUNC](-1)
        self.assertEqual(-0.7615941559557649, res)

    def test_gceq(self):
        test_concept = create_concept(activ_function='gceq')
        res = test_concept[NODE_ACTV_FUNC](0.7, 0.5)
        self.assertEqual(0.7, res)
        res = test_concept[NODE_ACTV_FUNC](7, 0.5)
        self.assertEqual(1.0, res)
        res = test_concept[NODE_ACTV_FUNC](0.3, 0.5)
        self.assertEqual(0.0, res)
        res = test_concept[NODE_ACTV_FUNC](-7, -3.5)
        self.assertEqual(0.0, res)
        res = test_concept[NODE_ACTV_FUNC](-2, -3.5)
        self.assertEqual(-1.0, res)

    def test_lceq(self):
        test_concept = create_concept(activ_function='lceq')
        res = test_concept[NODE_ACTV_FUNC](0.7, 0.5)
        self.assertEqual(0.0, res)
        res = test_concept[NODE_ACTV_FUNC](7, 0.5)
        self.assertEqual(0.0, res)
        res = test_concept[NODE_ACTV_FUNC](0.3, 0.5)
        self.assertEqual(0.3, res)
        res = test_concept[NODE_ACTV_FUNC](-7, -3.5)
        self.assertEqual(-1.0, res)
        res = test_concept[NODE_ACTV_FUNC](-2, -3.5)
        self.assertEqual(0.0, res)

    def test_fuzzy(self):
        test_concept = create_concept(activ_function='fuzzy')
        res = test_concept[NODE_ACTV_FUNC](10, np.array([0.0, 1.0]), np.array([5, 15]))
        self.assertEqual(0.5, res)
        res = test_concept[NODE_ACTV_FUNC](4, np.array([0.0, 1.0]), np.array([5, 15]))
        self.assertEqual(0.0, res)
        res = test_concept[NODE_ACTV_FUNC](16, np.array([0.0, 1.0]), np.array([5, 15]))
        self.assertEqual(1.0, res)
        res = test_concept[NODE_ACTV_FUNC](7, np.array([0.0, 0.5, 1.0]), np.array([5, 15, 20]))
        self.assertEqual(0.09999999999999998, res)
        res = test_concept[NODE_ACTV_FUNC](16, np.array([0.0, 0.5, 1.0]), np.array([5, 15, 20]))
        self.assertEqual(0.6, res)
        res = test_concept[NODE_ACTV_FUNC](-1.0, np.array([0.25, 0.5, 1.0, 0.5, 0.25]),
                                           np.array([0.2, 0.4, 0.6, 0.8, 1.0]))
        self.assertEqual(-0.75, res)
        res = test_concept[NODE_ACTV_FUNC](-0.5, np.array([0.25, 0.5, 1.0, 0.5, 0.25]),
                                           np.array([0.2, 0.4, 0.6, 0.8, 1.0]))
        self.assertEqual(-0.25, res)


class RelationFunctionsTests(unittest.TestCase):
    def test_supp(self):
        res = Relation.supp(1, 2, 2, 1)
        self.assertEqual(0.16666666666666666, res)

    def test_conf(self):
        res = Relation.conf(1, 1, 1, 1)
        self.assertEqual(0.5, res)

    def test_lift(self):
        res = Relation.lift(1, 2, 2, 1)
        self.assertEqual(0.6666666666666666, res)

    def test_odr(self):
        res = Relation.odr(1, 2, 2, 1)
        self.assertEqual(0.25, res)

    def test_rodr(self):
        res = Relation.rodr(1, 2, 2, 1)
        self.assertEqual(0.5, res)

    def test_pos_inf(self):
        res = Relation.pos_inf(1, 2, 2, 1)
        self.assertEqual(-0.3333333333333333, res)

    def test_simple(self):
        res = Relation.simple(1, 2, 2, 1)
        self.assertEqual(0.3333333333333333, res)


class OtherFucntionsTest(unittest.TestCase):
    def test_dual_quick_sort(self):
        array = np.array([2, 6, 1, 5])
        mirror = np.array([2, 6, 1, 5])
        dual_quick_sort(array, 0, len(array) - 1, mirror)
        for pos in range(len(array)):
            self.assertEqual(array[pos], mirror[pos])
