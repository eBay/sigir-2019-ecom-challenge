import evaluation_script
import unittest

class TestPredictions(unittest.TestCase):

    def test_g1_p1(self):
        expected = {'precision': 0.6, 'recall': 0.75, 'f1': 0.6666666666666665, 'tpr': 0.75, 'fpr': 0.6666666666666666}

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
        expected = {'precision': 0.5, 'recall': 1.0, 'f1': 0.6666666666666666, 'tpr': 1.0, 'fpr': 1.0}
        r = evaluation_script.evaluate("testfiles/ground_truth_set2.tsv",
                                       "testfiles/predictions_set2a.tsv", "supervised")
        rdata = r['result'][0]['data']
        for k, v in expected.items():
            self.assertAlmostEqual(v, rdata[k])

    def test_g2_p2b(self):
        expected = {'precision': 1, 'recall': 0.0, 'f1': 0.0, 'tpr': 0.0, 'fpr': 0.0}
        r = evaluation_script.evaluate("testfiles/ground_truth_set2.tsv",
                                       "testfiles/predictions_set2b.tsv", "supervised")
        rdata = r['result'][0]['data']
        for k, v in expected.items():
            self.assertAlmostEqual(v, rdata[k])

    def test_g2_p2c(self):
        expected = {'precision': 0.0, 'recall': 0.0, 'f1': 0, 'tpr': 0.0, 'fpr': 1.0}
        r = evaluation_script.evaluate("testfiles/ground_truth_set2.tsv",
                                       "testfiles/predictions_set2c.tsv", "supervised")
        rdata = r['result'][0]['data']
        for k, v in expected.items():
            self.assertAlmostEqual(v, rdata[k])

    def test_g2_p2d(self):
        expected = {'precision': 0.625, 'recall': 0.625, 'f1': 0.625, 'tpr': 0.625, 'fpr': 0.375}
        r = evaluation_script.evaluate("testfiles/ground_truth_set2.tsv",
                                       "testfiles/predictions_set2d.tsv", "supervised")
        rdata = r['result'][0]['data']
        for k, v in expected.items():
            self.assertAlmostEqual(v, rdata[k])

    def test_g2_p2e(self):
        expected = {'precision': 0.375, 'recall': 0.375, 'f1': 0.375, 'tpr': 0.375, 'fpr': 0.625}
        r = evaluation_script.evaluate("testfiles/ground_truth_set2.tsv",
                                       "testfiles/predictions_set2e.tsv", "supervised")
        rdata = r['result'][0]['data']
        for k, v in expected.items():
            self.assertAlmostEqual(v, rdata[k])

    def test_g3_p3a(self):
        expected = {'precision': 0.6923076923076923, 'recall': 0.75, 'f1': 0.7199999999999999,
                    'tpr': 0.75, 'fpr': 0.34285714285714286}
        r = evaluation_script.evaluate("testfiles/ground_truth_set3.tsv",
                                       "testfiles/predictions_set3a.tsv", "supervised")
        rdata = r['result'][0]['data']
        for k, v in expected.items():
            self.assertAlmostEqual(v, rdata[k])

    def test_g3_p3b(self):
        expected = {'precision': 0.5333333333333333, 'recall': 0.4444444444444444, 'f1': 0.4848484848484848,
                    'tpr': 0.4444444444444444, 'fpr': 0.4}
        r = evaluation_script.evaluate("testfiles/ground_truth_set3.tsv",
                                       "testfiles/predictions_set3b.tsv", "supervised")
        rdata = r['result'][0]['data']
        for k, v in expected.items():
            self.assertAlmostEqual(v, rdata[k])

    def test_g3_p3c(self):
        expected = {'precision': 0.5675675675675675, 'recall': 0.5833333333333334, 'f1': 0.5753424657534246,
                    'tpr': 0.5833333333333334, 'fpr': 0.45714285714285713}
        r = evaluation_script.evaluate("testfiles/ground_truth_set3.tsv",
                                       "testfiles/predictions_set3c.tsv", "supervised")
        rdata = r['result'][0]['data']
        for k, v in expected.items():
            self.assertAlmostEqual(v, rdata[k])

            
if __name__ == '__main__':
    unittest.main()
