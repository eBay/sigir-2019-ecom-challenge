import evaluation_script
import unittest

class TestPredictions(unittest.TestCase):

    def test_g1_p1(self):
        expected = {'global_precision': 0.6,
                    'global_recall': 0.75, 
                    'global_f1': 0.6666666666666665,
                    'global_tpr': 0.75,
                    'global_fpr': 0.6666666666666666,
                    'global_accuracy': 0.5714285714285714,
                    'average_precision': 0.7777777777777777,
                    'average_recall': 0.8333333333333334,
                    'average_f1': 0.7222222222222222,
                    'average_tpr': 0.8333333333333334,
                    'average_fpr': 0.6666666666666666,
                    'average_accuracy': 0.611111111111111,
                    'average_l2h_ndcg@10': 0.7169361380260636,
                    'average_h2l_ndcg@10': 0.7959842760619721,
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
            if k != 'average_l2h_ndcg@10' and k != 'average_h2l_ndcg@10':
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
            if k != 'average_l2h_ndcg@10' and k != 'average_h2l_ndcg@10':
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
            if k != 'average_l2h_ndcg@10' and k != 'average_h2l_ndcg@10':
                self.assertAlmostEqual(v, rdata[k])
                self.assertAlmostEqual(v, rdata2[k])
            self.assertAlmostEqual(v, rdata3[k])

    def test_g2_p2a(self):
        expected = {'global_precision': 0.5,
                    'global_recall': 1.0,
                    'global_f1': 0.6666666666666666,
                    'global_tpr': 1.0,
                    'global_fpr': 1.0,
                    'global_accuracy': 0.5,
                    'average_precision': 0.5,
                    'average_recall': 1.0,
                    'average_f1': 0.619047619,
                    'average_tpr': 1.0,
                    'average_fpr': 1.0,
                    'average_accuracy': 0.5,
                    'average_l2h_ndcg@10': 0.9971792416440344,
                    'average_h2l_ndcg@10': 0.6830002811190978,
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
            if k != 'average_l2h_ndcg@10' and k != 'average_h2l_ndcg@10':
                self.assertAlmostEqual(v, rdata[k])
                self.assertAlmostEqual(v, rdata2[k])
            self.assertAlmostEqual(v, rdata3[k])

    def test_g2_p2b(self):
        expected = {'global_precision': 1,
                    'global_recall': 0.0,
                    'global_f1': 0.0,
                    'global_tpr': 0.0,
                    'global_fpr': 0.0,
                    'global_accuracy': 0.5,
                    'average_precision': 1.0,
                    'average_recall': 0.142857143,
                    'average_f1': 0.0,
                    'average_tpr': 0.142857143,
                    'average_fpr': 0.142857143,
                    'average_accuracy': 0.5,
                    'average_l2h_ndcg@10': 0.14285714285714285,
                    'average_h2l_ndcg@10': 0.14285714285714285,
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
            if k != 'average_l2h_ndcg@10' and k != 'average_h2l_ndcg@10':
                self.assertAlmostEqual(v, rdata[k])
                self.assertAlmostEqual(v, rdata2[k])
            self.assertAlmostEqual(v, rdata3[k])

    def test_g2_p2c(self):
        expected = {'global_precision': 0.0,
                    'global_recall': 0.0,
                    'global_f1': 0,
                    'global_tpr': 0.0,
                    'global_fpr': 1.0,
                    'global_accuracy': 0.0,
                    'average_precision': 0.142857143,
                    'average_recall': 0.142857143,
                    'average_f1': 0.0,
                    'average_tpr': 0.142857143,
                    'average_fpr': 1.0,
                    'average_accuracy': 0.0,
                    'average_l2h_ndcg@10': 0.9971792416440344,
                    'average_h2l_ndcg@10': 0.6830002811190978,
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
            if k != 'average_l2h_ndcg@10' and k != 'average_h2l_ndcg@10':
                self.assertAlmostEqual(v, rdata[k])
                self.assertAlmostEqual(v, rdata2[k])
            self.assertAlmostEqual(v, rdata3[k])

    def test_g2_p2d(self):
        expected = {'global_precision': 0.625,
                    'global_recall': 0.625,
                    'global_f1': 0.625,
                    'global_tpr': 0.625,
                    'global_fpr': 0.375,
                    'global_accuracy': 0.625,
                    'average_precision': 0.6666666666666666,
                    'average_recall': 0.714285714,
                    'average_f1': 0.585714286,
                    'average_tpr': 0.714285714,
                    'average_fpr': 0.428571429,
                    'average_accuracy': .6547619047619048,
                    'average_l2h_ndcg@10': 0.8070645353018062,
                    'average_h2l_ndcg@10': 0.6555242102166945,
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
            if k != 'average_l2h_ndcg@10' and k != 'average_h2l_ndcg@10':
                self.assertAlmostEqual(v, rdata[k])
                self.assertAlmostEqual(v, rdata2[k])
            self.assertAlmostEqual(v, rdata3[k])

    def test_g2_p2e(self):
        expected = {'global_precision': 0.375,
                    'global_recall': 0.375,
                    'global_f1': 0.375,
                    'global_tpr': 0.375,
                    'global_fpr': 0.625,
                    'global_accuracy': 0.375,
                    'average_precision': 0.342857143,
                    'average_recall': 0.428571429,
                    'average_f1': 0.285714286,
                    'average_tpr': 0.428571429,
                    'average_fpr': 0.714285714,
                    'average_accuracy': 0.34523809523809523,
                    'average_l2h_ndcg@10': 0.36866848828077303,
                    'average_h2l_ndcg@10': 0.4462728422747134,
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
            if k != 'average_l2h_ndcg@10' and k != 'average_h2l_ndcg@10':
                self.assertAlmostEqual(v, rdata[k])
                self.assertAlmostEqual(v, rdata2[k])
            self.assertAlmostEqual(v, rdata3[k])

    def test_g3_p3a(self):
        expected = {'global_precision': 0.6923076923076923,
                    'global_recall': 0.75,
                    'global_f1': 0.7199999999999999,
                    'global_tpr': 0.75,
                    'global_fpr': 0.34285714285714286,
                    'global_accuracy': 0.704225352112676,
                    'average_precision': 0.7517857142857143,
                    'average_recall': 0.7547619047619047,
                    'average_f1': 0.7340659340659341,
                    'average_tpr': 0.7547619047619047,
                    'average_fpr': 0.3625,
                    'average_accuracy': 0.7050369769119769,
                    'average_l2h_ndcg@10': 0.5995043313788928,
                    'average_h2l_ndcg@10': 0.8314918192086054,
        }

        r = evaluation_script.evaluate("testfiles/ground_truth_set3.tsv",
                                       "testfiles/predictions_set3a.tsv", "supervised")
        rdata = r['result'][0]['data']
        rdata2 = evaluation_script.evaluate_submission("testfiles/ground_truth_set3.tsv",
                                                       "testfiles/predictions_set3a.tsv")
        rdata3 = evaluation_script.evaluate_submission("testfiles/ground_truth_set3.tsv",
                                                       "testfiles/predictions_set3a.tsv",
                                                       "testfiles/documents_set3.tsv")
        for k, v in expected.items():
            if k != 'average_l2h_ndcg@10' and k != 'average_h2l_ndcg@10':
                self.assertAlmostEqual(v, rdata[k])
                self.assertAlmostEqual(v, rdata2[k])
            self.assertAlmostEqual(v, rdata3[k])

    def test_g3_p3b(self):
        expected = {'global_precision': 0.5333333333333333,
                    'global_recall': 0.4444444444444444,
                    'global_f1': 0.4848484848484848,
                    'global_tpr': 0.4444444444444444,
                    'global_fpr': 0.4,
                    'global_accuracy': 0.5211267605633803,
                    'average_precision': 0.5708333333333334,
                    'average_recall': 0.4857142857142857,
                    'average_f1': 0.4978174603174603,
                    'average_tpr': 0.4857142857142857,
                    'average_fpr': 0.39166666666666666,
                    'average_accuracy': 0.5315205627705628,
                    'average_l2h_ndcg@10': 0.3287689269825538,
                    'average_h2l_ndcg@10': 0.5639633827558276,
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
            if k != 'average_l2h_ndcg@10' and k != 'average_h2l_ndcg@10':
                self.assertAlmostEqual(v, rdata[k])
                self.assertAlmostEqual(v, rdata2[k])
            self.assertAlmostEqual(v, rdata3[k])

    def test_g3_p3c(self):
        expected = {'global_precision': 0.5675675675675675,
                    'global_recall': 0.5833333333333334,
                    'global_f1': 0.5753424657534246,
                    'global_tpr': 0.5833333333333334,
                    'global_fpr': 0.45714285714285713,
                    'global_accuracy': 0.5633802816901409,
                    'average_precision': 0.5708333333333333,
                    'average_recall': 0.5851190476190475,
                    'average_f1': 0.5680826118326119,
                    'average_tpr': 0.5851190476190475,
                    'average_fpr': 0.4125,
                    'average_accuracy': 0.5734397546897546,
                    'average_l2h_ndcg@10': 0.561254350448992,
                    'average_h2l_ndcg@10': 0.5824553461136641,
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
            if k != 'average_l2h_ndcg@10' and k != 'average_h2l_ndcg@10':
                self.assertAlmostEqual(v, rdata[k])
                self.assertAlmostEqual(v, rdata2[k])
            self.assertAlmostEqual(v, rdata3[k])

            
if __name__ == '__main__':
    unittest.main()
