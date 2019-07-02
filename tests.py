import evaluation_script
import unittest
import shutil
import os

class TestPredictions(unittest.TestCase):

    def test_g1_p1(self):
        expected = {'precision': 0.6,
                    'recall': 0.75,
                    'f1': 0.6666666666666665,
                    'tpr': 0.75,
                    'fpr': 0.6666666666666666,
                    'accuracy': 0.5714285714285714,
                    'ave_precision': 0.7777777777777777,
                    'ave_recall': 0.8333333333333334,
                    'ave_f1': 0.7222222222222222,
                    'ave_tpr': 0.8333333333333334,
                    'ave_fpr': 0.6666666666666666,
                    'ave_accuracy': 0.611111111111111,
                    'l2h_ndcg10': 0.7169361380260636,
                    'h2l_ndcg10': 0.7959842760619721,
        }

        r = evaluation_script.evaluate("testfiles/ground_truth_set1.tsv",
                                       "testfiles/predictions_set1a.tsv", "supervised")
        rdata = r['result'][0]['data']
        rdata2 = evaluation_script.evaluate_submission("testfiles/ground_truth_set1.tsv",
                                                       "testfiles/predictions_set1a.tsv")
        rdata3 = evaluation_script.evaluate_submission("testfiles/ground_truth_set1.tsv",
                                                       "testfiles/predictions_set1a.tsv",
                                                       "testfiles/documents_set1.tsv")
        for k, v in expected.items():
            if k != 'l2h_ndcg10' and k != 'h2l_ndcg10':
                self.assertAlmostEqual(v, rdata[k])
                self.assertAlmostEqual(v, rdata2[k])
            self.assertAlmostEqual(v, rdata3[k])

        r = evaluation_script.evaluate("testfiles/ground_truth_set1.tsv",
                                       "testfiles/predictions_set1b.tsv", "supervised")
        rdata = r['result'][0]['data']
        rdata2 = evaluation_script.evaluate_submission("testfiles/ground_truth_set1.tsv",
                                                       "testfiles/predictions_set1b.tsv")
        rdata3 = evaluation_script.evaluate_submission("testfiles/ground_truth_set1.tsv",
                                                       "testfiles/predictions_set1b.tsv",
                                                       "testfiles/documents_set1.tsv")
        for k, v in expected.items():
            if k != 'l2h_ndcg10' and k != 'h2l_ndcg10':
                self.assertAlmostEqual(v, rdata[k])
                self.assertAlmostEqual(v, rdata2[k])
            self.assertAlmostEqual(v, rdata3[k])

        r = evaluation_script.evaluate("testfiles/ground_truth_set1_compressed.tsv.gz",
                                       "testfiles/predictions_set1a_compressed.tsv.gz",
                                       "supervised")
        rdata = r['result'][0]['data']
        rdata2 = evaluation_script.evaluate_submission("testfiles/ground_truth_set1_compressed.tsv.gz",
                                                       "testfiles/predictions_set1a_compressed.tsv.gz")
        rdata3 = evaluation_script.evaluate_submission("testfiles/ground_truth_set1_compressed.tsv.gz",
                                                       "testfiles/predictions_set1a_compressed.tsv.gz",
                                                       "testfiles/documents_set1_compressed.tsv.gz")
        for k, v in expected.items():
            if k != 'l2h_ndcg10' and k != 'h2l_ndcg10':
                self.assertAlmostEqual(v, rdata[k])
                self.assertAlmostEqual(v, rdata2[k])
            self.assertAlmostEqual(v, rdata3[k])

    def test_g2_p2a(self):
        expected = {'precision': 0.5,
                    'recall': 1.0,
                    'f1': 0.6666666666666666,
                    'tpr': 1.0,
                    'fpr': 1.0,
                    'accuracy': 0.5,
                    'ave_precision': 0.5,
                    'ave_recall': 1.0,
                    'ave_f1': 0.619047619,
                    'ave_tpr': 1.0,
                    'ave_fpr': 1.0,
                    'ave_accuracy': 0.5,
                    'l2h_ndcg10': 0.9971792416440344,
                    'h2l_ndcg10': 0.6830002811190978,
        }

        r = evaluation_script.evaluate("testfiles/ground_truth_set2.tsv",
                                       "testfiles/predictions_set2a.tsv", "supervised")
        rdata = r['result'][0]['data']
        rdata2 = evaluation_script.evaluate_submission("testfiles/ground_truth_set2.tsv",
                                                       "testfiles/predictions_set2a.tsv")
        rdata3 = evaluation_script.evaluate_submission("testfiles/ground_truth_set2.tsv",
                                                       "testfiles/predictions_set2a.tsv",
                                                       "testfiles/documents_set2.tsv")
        for k, v in expected.items():
            if k != 'l2h_ndcg10' and k != 'h2l_ndcg10':
                self.assertAlmostEqual(v, rdata[k])
                self.assertAlmostEqual(v, rdata2[k])
            self.assertAlmostEqual(v, rdata3[k])

    def test_g2_p2b(self):
        expected = {'precision': 1,
                    'recall': 0.0,
                    'f1': 0.0,
                    'tpr': 0.0,
                    'fpr': 0.0,
                    'accuracy': 0.5,
                    'ave_precision': 1.0,
                    'ave_recall': 0.142857143,
                    'ave_f1': 0.0,
                    'ave_tpr': 0.142857143,
                    'ave_fpr': 0.142857143,
                    'ave_accuracy': 0.5,
                    'l2h_ndcg10': 0.14285714285714285,
                    'h2l_ndcg10': 0.14285714285714285,
        }

        r = evaluation_script.evaluate("testfiles/ground_truth_set2.tsv",
                                       "testfiles/predictions_set2b.tsv", "supervised")
        rdata = r['result'][0]['data']
        rdata2 = evaluation_script.evaluate_submission("testfiles/ground_truth_set2.tsv",
                                                       "testfiles/predictions_set2b.tsv")
        rdata3 = evaluation_script.evaluate_submission("testfiles/ground_truth_set2.tsv",
                                                       "testfiles/predictions_set2b.tsv",
                                                       "testfiles/documents_set2.tsv")
        for k, v in expected.items():
            if k != 'l2h_ndcg10' and k != 'h2l_ndcg10':
                self.assertAlmostEqual(v, rdata[k])
                self.assertAlmostEqual(v, rdata2[k])
            self.assertAlmostEqual(v, rdata3[k])

    def test_g2_p2c(self):
        expected = {'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0,
                    'tpr': 0.0,
                    'fpr': 1.0,
                    'accuracy': 0.0,
                    'ave_precision': 0.142857143,
                    'ave_recall': 0.142857143,
                    'ave_f1': 0.0,
                    'ave_tpr': 0.142857143,
                    'ave_fpr': 1.0,
                    'ave_accuracy': 0.0,
                    'l2h_ndcg10': 0.9971792416440344,
                    'h2l_ndcg10': 0.6830002811190978,
        }

        r = evaluation_script.evaluate("testfiles/ground_truth_set2.tsv",
                                       "testfiles/predictions_set2c.tsv", "supervised")
        rdata = r['result'][0]['data']
        rdata2 = evaluation_script.evaluate_submission("testfiles/ground_truth_set2.tsv",
                                                       "testfiles/predictions_set2c.tsv")
        rdata3 = evaluation_script.evaluate_submission("testfiles/ground_truth_set2.tsv",
                                                       "testfiles/predictions_set2c.tsv",
                                                       "testfiles/documents_set2.tsv")
        for k, v in expected.items():
            if k != 'l2h_ndcg10' and k != 'h2l_ndcg10':
                self.assertAlmostEqual(v, rdata[k])
                self.assertAlmostEqual(v, rdata2[k])
            self.assertAlmostEqual(v, rdata3[k])

    def test_g2_p2d(self):
        expected = {'precision': 0.625,
                    'recall': 0.625,
                    'f1': 0.625,
                    'tpr': 0.625,
                    'fpr': 0.375,
                    'accuracy': 0.625,
                    'ave_precision': 0.6666666666666666,
                    'ave_recall': 0.714285714,
                    'ave_f1': 0.585714286,
                    'ave_tpr': 0.714285714,
                    'ave_fpr': 0.428571429,
                    'ave_accuracy': .6547619047619048,
                    'l2h_ndcg10': 0.8070645353018062,
                    'h2l_ndcg10': 0.6555242102166945,
        }

        r = evaluation_script.evaluate("testfiles/ground_truth_set2.tsv",
                                       "testfiles/predictions_set2d.tsv", "supervised")
        rdata = r['result'][0]['data']
        rdata2 = evaluation_script.evaluate_submission("testfiles/ground_truth_set2.tsv",
                                                       "testfiles/predictions_set2d.tsv")
        rdata3 = evaluation_script.evaluate_submission("testfiles/ground_truth_set2.tsv",
                                                       "testfiles/predictions_set2d.tsv",
                                                       "testfiles/documents_set2.tsv")
        for k, v in expected.items():
            if k != 'l2h_ndcg10' and k != 'h2l_ndcg10':
                self.assertAlmostEqual(v, rdata[k])
                self.assertAlmostEqual(v, rdata2[k])
            self.assertAlmostEqual(v, rdata3[k])

    def test_g2_p2e(self):
        expected = {'precision': 0.375,
                    'recall': 0.375,
                    'f1': 0.375,
                    'tpr': 0.375,
                    'fpr': 0.625,
                    'accuracy': 0.375,
                    'ave_precision': 0.342857143,
                    'ave_recall': 0.428571429,
                    'ave_f1': 0.285714286,
                    'ave_tpr': 0.428571429,
                    'ave_fpr': 0.714285714,
                    'ave_accuracy': 0.34523809523809523,
                    'l2h_ndcg10': 0.36866848828077303,
                    'h2l_ndcg10': 0.4462728422747134,
        }

        r = evaluation_script.evaluate("testfiles/ground_truth_set2.tsv",
                                       "testfiles/predictions_set2e.tsv", "supervised")
        rdata = r['result'][0]['data']
        rdata2 = evaluation_script.evaluate_submission("testfiles/ground_truth_set2.tsv",
                                                       "testfiles/predictions_set2e.tsv")
        rdata3 = evaluation_script.evaluate_submission("testfiles/ground_truth_set2.tsv",
                                                       "testfiles/predictions_set2e.tsv",
                                                       "testfiles/documents_set2.tsv")
        for k, v in expected.items():
            if k != 'l2h_ndcg10' and k != 'h2l_ndcg10':
                self.assertAlmostEqual(v, rdata[k])
                self.assertAlmostEqual(v, rdata2[k])
            self.assertAlmostEqual(v, rdata3[k])

    def test_g3_p3a(self):
        expected = {'precision': 0.6923076923076923,
                    'recall': 0.75,
                    'f1': 0.7199999999999999,
                    'tpr': 0.75,
                    'fpr': 0.34285714285714286,
                    'accuracy': 0.704225352112676,
                    'ave_precision': 0.7517857142857143,
                    'ave_recall': 0.7547619047619047,
                    'ave_f1': 0.7340659340659341,
                    'ave_tpr': 0.7547619047619047,
                    'ave_fpr': 0.3625,
                    'ave_accuracy': 0.7050369769119769,
                    'l2h_ndcg10': 0.5995043313788928,
                    'h2l_ndcg10': 0.8314918192086054,
        }

        r = evaluation_script.evaluate("testfiles/ground_truth_set3.tsv",
                                       "testfiles/predictions_set3a.tsv", "supervised")
        rdata = r['result'][0]['data']
        rdata2 = evaluation_script.evaluate_submission("testfiles/ground_truth_set3.tsv",
                                                       "testfiles/predictions_set3a.tsv")
        rdata3 = evaluation_script.evaluate_submission("testfiles/ground_truth_set3.tsv",
                                                       "testfiles/predictions_set3a.tsv",
                                                       "testfiles/documents_set3.tsv")

        # Data sets 4 and 5 test the evaluate function reading from the default document
        # file when present in the directory of the evaluation_script package.
        shutil.copy2('testfiles2/documents.tsv', 'evaluation_script')
        r4 = evaluation_script.evaluate("testfiles2/ground_truth_set3.tsv",
                                        "testfiles2/predictions_set3a.tsv", "supervised")
        rdata4 = r4['result'][0]['data']
        os.remove('evaluation_script/documents.tsv')
        

        shutil.copy2('testfiles3/documents.tsv.gz', 'evaluation_script')
        r5 = evaluation_script.evaluate("testfiles3/ground_truth_set3.tsv.gz",
                                        "testfiles3/predictions_set3a.tsv.gz", "supervised")
        rdata5 = r5['result'][0]['data']
        os.remove('evaluation_script/documents.tsv.gz')


        for k, v in expected.items():
            if k != 'l2h_ndcg10' and k != 'h2l_ndcg10':
                self.assertAlmostEqual(v, rdata[k])
                self.assertAlmostEqual(v, rdata2[k])
            self.assertAlmostEqual(v, rdata3[k])
            self.assertAlmostEqual(v, rdata4[k])
            self.assertAlmostEqual(v, rdata5[k])

    def test_g3_p3b(self):
        expected = {'precision': 0.5333333333333333,
                    'recall': 0.4444444444444444,
                    'f1': 0.4848484848484848,
                    'tpr': 0.4444444444444444,
                    'fpr': 0.4,
                    'accuracy': 0.5211267605633803,
                    'ave_precision': 0.5708333333333334,
                    'ave_recall': 0.4857142857142857,
                    'ave_f1': 0.4978174603174603,
                    'ave_tpr': 0.4857142857142857,
                    'ave_fpr': 0.39166666666666666,
                    'ave_accuracy': 0.5315205627705628,
                    'l2h_ndcg10': 0.3287689269825538,
                    'h2l_ndcg10': 0.5639633827558276,
        }

        r = evaluation_script.evaluate("testfiles/ground_truth_set3.tsv",
                                       "testfiles/predictions_set3b.tsv", "supervised")
        rdata = r['result'][0]['data']
        rdata2 = evaluation_script.evaluate_submission("testfiles/ground_truth_set3.tsv",
                                                       "testfiles/predictions_set3b.tsv")
        rdata3 = evaluation_script.evaluate_submission("testfiles/ground_truth_set3.tsv",
                                                       "testfiles/predictions_set3b.tsv",
                                                       "testfiles/documents_set3.tsv")
        for k, v in expected.items():
            if k != 'l2h_ndcg10' and k != 'h2l_ndcg10':
                self.assertAlmostEqual(v, rdata[k])
                self.assertAlmostEqual(v, rdata2[k])
            self.assertAlmostEqual(v, rdata3[k])

    def test_g3_p3c(self):
        expected = {'precision': 0.5675675675675675,
                    'recall': 0.5833333333333334,
                    'f1': 0.5753424657534246,
                    'tpr': 0.5833333333333334,
                    'fpr': 0.45714285714285713,
                    'accuracy': 0.5633802816901409,
                    'ave_precision': 0.5708333333333333,
                    'ave_recall': 0.5851190476190475,
                    'ave_f1': 0.5680826118326119,
                    'ave_tpr': 0.5851190476190475,
                    'ave_fpr': 0.4125,
                    'ave_accuracy': 0.5734397546897546,
                    'l2h_ndcg10': 0.561254350448992,
                    'h2l_ndcg10': 0.5824553461136641,
        }

        r = evaluation_script.evaluate("testfiles/ground_truth_set3.tsv",
                                       "testfiles/predictions_set3c.tsv", "supervised")
        rdata = r['result'][0]['data']
        rdata2 = evaluation_script.evaluate_submission("testfiles/ground_truth_set3.tsv",
                                                       "testfiles/predictions_set3c.tsv")
        rdata3 = evaluation_script.evaluate_submission("testfiles/ground_truth_set3.tsv",
                                                       "testfiles/predictions_set3c.tsv",
                                                       "testfiles/documents_set3.tsv")
        for k, v in expected.items():
            if k != 'l2h_ndcg10' and k != 'h2l_ndcg10':
                self.assertAlmostEqual(v, rdata[k])
                self.assertAlmostEqual(v, rdata2[k])
            self.assertAlmostEqual(v, rdata3[k])


if __name__ == '__main__':
    unittest.main()
