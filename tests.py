import evaluation_script
import unittest

class TestPredictions(unittest.TestCase):

    def test_g1_p1(self):
        expected = {'global_precision': 0.6,
                    'global_recall': 0.75, 
                    'global_f1': 0.6666666666666665,
                    'global_tpr': 0.75,
                    'global_fpr': 0.6666666666666666,
                    'averaged_precision': 0.7777777777777777,
                    'averaged_recall': 0.8333333333333334,
                    'averaged_f1': 0.7222222222222222,
                    'averaged_tpr': 0.8333333333333334,
                    'averaged_fpr': 0.6666666666666666}

        r = evaluation_script.evaluate("testfiles/ground_truth_set1.tsv",
                                       "testfiles/predictions_set1a.tsv", "supervised")
        rdata = r['result'][0]['data']
        for k, v in expected.items():
            self.assertAlmostEqual(v, rdata[k])

        r = evaluation_script.evaluate("testfiles/ground_truth_set1.tsv",
                                       "testfiles/predictions_set1b.tsv", "supervised")
        rdata = r['result'][0]['data']
        for k, v in expected.items():
            self.assertAlmostEqual(v, rdata[k])

        r = evaluation_script.evaluate("testfiles/ground_truth_set1_compressed.tsv.gz",
                                       "testfiles/predictions_set1a_compressed.tsv.gz",
                                       "supervised")
        rdata = r['result'][0]['data']
        for k, v in expected.items():
            self.assertAlmostEqual(v, rdata[k])

    def test_g2_p2a(self):
        expected = {'global_precision': 0.5,
                    'global_recall': 1.0,
                    'global_f1': 0.6666666666666666,
                    'global_tpr': 1.0,
                    'global_fpr': 1.0,
                    'averaged_precision': 0.4375,
                    'averaged_recall': 0.875,
                    'averaged_f1': 0.5416666666666666,
                    'averaged_tpr': 0.875,
                    'averaged_fpr': 0.875}
        r = evaluation_script.evaluate("testfiles/ground_truth_set2.tsv",
                                       "testfiles/predictions_set2a.tsv", "supervised")
        rdata = r['result'][0]['data']
        for k, v in expected.items():
            self.assertAlmostEqual(v, rdata[k])

    def test_g2_p2b(self):
        expected = {'global_precision': 1,
                    'global_recall': 0.0,
                    'global_f1': 0.0,
                    'global_tpr': 0.0,
                    'global_fpr': 0.0,
                    'averaged_precision': 0.875,
                    'averaged_recall': 0.125,
                    'averaged_f1': 0.0,
                    'averaged_tpr': 0.125,
                    'averaged_fpr': 0.125}
        r = evaluation_script.evaluate("testfiles/ground_truth_set2.tsv",
                                       "testfiles/predictions_set2b.tsv", "supervised")
        rdata = r['result'][0]['data']
        for k, v in expected.items():
            self.assertAlmostEqual(v, rdata[k])

    def test_g2_p2c(self):
        expected = {'global_precision': 0.0,
                    'global_recall': 0.0,
                    'global_f1': 0,
                    'global_tpr': 0.0,
                    'global_fpr': 1.0,
                    'averaged_precision': 0.125,
                    'averaged_recall': 0.125,
                    'averaged_f1': 0.0,
                    'averaged_tpr': 0.125,
                    'averaged_fpr': 0.875}
        r = evaluation_script.evaluate("testfiles/ground_truth_set2.tsv",
                                       "testfiles/predictions_set2c.tsv", "supervised")
        rdata = r['result'][0]['data']
        for k, v in expected.items():
            self.assertAlmostEqual(v, rdata[k])

    def test_g2_p2d(self):
        expected = {'global_precision': 0.625,
                    'global_recall': 0.625,
                    'global_f1': 0.625,
                    'global_tpr': 0.625,
                    'global_fpr': 0.375,
                    'averaged_precision': 0.5833333333333333,
                    'averaged_recall': 0.625,
                    'averaged_f1': 0.5125,
                    'averaged_tpr': 0.625,
                    'averaged_fpr': 0.375}
        r = evaluation_script.evaluate("testfiles/ground_truth_set2.tsv",
                                       "testfiles/predictions_set2d.tsv", "supervised")
        rdata = r['result'][0]['data']
        for k, v in expected.items():
            self.assertAlmostEqual(v, rdata[k])

    def test_g2_p2e(self):
        expected = {'global_precision': 0.375,
                    'global_recall': 0.375,
                    'global_f1': 0.375,
                    'global_tpr': 0.375,
                    'global_fpr': 0.625,
                    'averaged_precision': 0.3,
                    'averaged_recall': 0.375,
                    'averaged_f1': 0.25,
                    'averaged_tpr': 0.375,
                    'averaged_fpr': 0.625}
        r = evaluation_script.evaluate("testfiles/ground_truth_set2.tsv",
                                       "testfiles/predictions_set2e.tsv", "supervised")
        rdata = r['result'][0]['data']
        for k, v in expected.items():
            self.assertAlmostEqual(v, rdata[k])

    def test_g3_p3a(self):
        expected = {'global_precision': 0.6923076923076923,
                    'global_recall': 0.75,
                    'global_f1': 0.7199999999999999,
                    'global_tpr': 0.75,
                    'global_fpr': 0.34285714285714286,
                    'averaged_precision': 0.7517857142857143,
                    'averaged_recall': 0.7547619047619047,
                    'averaged_f1': 0.7340659340659341,
                    'averaged_tpr': 0.7547619047619047,
                    'averaged_fpr': 0.3625}
        r = evaluation_script.evaluate("testfiles/ground_truth_set3.tsv",
                                       "testfiles/predictions_set3a.tsv", "supervised")
        rdata = r['result'][0]['data']
        for k, v in expected.items():
            self.assertAlmostEqual(v, rdata[k])

    def test_g3_p3b(self):
        expected = {'global_precision': 0.5333333333333333,
                    'global_recall': 0.4444444444444444,
                    'global_f1': 0.4848484848484848,
                    'global_tpr': 0.4444444444444444,
                    'global_fpr': 0.4,
                    'averaged_precision': 0.5708333333333334,
                    'averaged_recall': 0.4857142857142857,
                    'averaged_f1': 0.4978174603174603,
                    'averaged_tpr': 0.4857142857142857,
                    'averaged_fpr': 0.39166666666666666}
        r = evaluation_script.evaluate("testfiles/ground_truth_set3.tsv",
                                       "testfiles/predictions_set3b.tsv", "supervised")
        rdata = r['result'][0]['data']
        for k, v in expected.items():
            self.assertAlmostEqual(v, rdata[k])

    def test_g3_p3c(self):
        expected = {'global_precision': 0.5675675675675675,
                    'global_recall': 0.5833333333333334,
                    'global_f1': 0.5753424657534246,
                    'global_tpr': 0.5833333333333334,
                    'global_fpr': 0.45714285714285713,
                    'averaged_precision': 0.5708333333333333,
                    'averaged_recall': 0.5851190476190475,
                    'averaged_f1': 0.5680826118326119,
                    'averaged_tpr': 0.5851190476190475,
                    'averaged_fpr': 0.4125}
        r = evaluation_script.evaluate("testfiles/ground_truth_set3.tsv",
                                       "testfiles/predictions_set3c.tsv", "supervised")
        rdata = r['result'][0]['data']
        for k, v in expected.items():
            self.assertAlmostEqual(v, rdata[k])

            
if __name__ == '__main__':
    unittest.main()
