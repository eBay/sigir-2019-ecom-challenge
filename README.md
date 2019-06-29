# eBay SIGIR 2019 eCommerce Search Challenge &nbsp; [![Travis](https://img.shields.io/travis/eBay/sigir-2019-ecom-challenge.svg)](https://travis-ci.com/eBay/sigir-2019-ecom-challenge)

This repository contains auxiliary tools used to support the [eBay SIGIR 2019 eCommerce Search Challenge: High Accuracy Recall Task](https://sigir-ecom.github.io/data-task.html). Currently this repository consists of one tool: An evaluation script challenge participants can use to score their predictions locally.

More information about the challenge can be found on the [SIGIR eCom'19 Data Challenge](https://sigir-ecom.github.io/data-task.html) page.

## Evaluation Script (eval_predictions.py)

The script provided here is a command line version of the evaluation script used on the [EvalAI](https://sigir-ecom.github.io/data-task.html) challenge hosting site to score participant submissions. The script allows participants to run the scoring function on their own train and test data sets. Other than having a command line interface it is the same as the run on the evalAI server.

The script takes two files, a "ground truth" file and a "prediction" file. These are tab-separated-value files in the same format used for challenge submissions. An example run:

```
$ python3 eval_predictions.py --ground-truth-file ground_truth.tsv  --prediction-file predictions.tsv

{'precision': 0.7631944059048782, 'recall': 0.7421078193297718, 'f1': 0.7525034199726401, 'tpr': 0.7421078193297718, 'fpr': 0.16721529900462417, 'accuracy': 0.7946377897341597, 'ave_precision': 0.7682185259671204, 'ave_recall': 0.6742832851232685, 'ave_f1': 0.6500524267152122, 'ave_tpr': 0.6742832851232685, 'ave_fpr': 0.20725801902745247, 'ave_accuracy': 0.7948493212347685, 'l2h_ndcg10': 0.0, 'h2l_ndcg10': 0.0}
```

The final output line has the same result values as used on the [challenge leaderboard](https://evalai.cloudcv.org/web/challenges/challenge-page/361/leaderboard).

The script has been tested with both python 2 and python 3. The short form options `-g` and `-p` can be used in place of the longer `--ground-truth-file` and `--prediction-file` form options. Both plain text and gzipped files are supported. (Please use `.tsv` or `.gz` file extensions.) An example call with short options and gzipped files:

```
$ python3 eval_predictions.py -g ground_truth.tsv.gz -p predictions.tsv.gz
```

The script can also calculate price-based NDCG scores. The "documents" file from the challenge data is needed for this. Use the `--d|--document-file` option. For example:
```
$ python3 eval_predictions.py -g ground_truth.tsv.gz -p predictions.tsv.gz -d documents.tsv.gz
```

An example showing the file format is given below. Challenge participants will find more information in the `readme` file distributed with the data files.

### Evaluation script example

This example uses a couple of small files to show how the script works. The files are found in `samples/` directory. Data in the files is shown in the tables below:

`ground_truth_set1.tsv`

| doc/query |  1 |  2 |  3 |
| --------- | -- | -- | -- |
| 101       |  1 |  0 |  0 |
| 102       |  0 | -1 |  1 |
| 103       |  0 |  1 | -1 |
| 104       |  1 |  0 | -1 |

`predictions.tsv`

| doc/query |  1 |  2 |  3 |
| --------- | -- | -- | -- |
| 101       | -1 | -1 | -1 |
| 102       |  1 | -1 |  1 |
| 103       |  1 |  1 |  1 |
| 104       |  1 | -1 |  1 |

The `ground_truth.tsv` file has labels for seven of the twelve query-document pairs (those with `1` or `-1` entries). The `predictions.tsv` file has predictions for all twelve of the query-document pairs. The `eval_predictions.py` script will generate a score based on the seven query-document pairs having relevant-or-not labels.

The basic metrics calculations for these files:

| Metric                    | Value | Calculation |
| ------------------------- | ----- | ----------- |
| True Positives (tp)       | 3     |             |
| True Negatives (tn)       | 1     |             |
| False Positives (fp)      | 2     |             |
| False Negatives (fn)      | 1     |             |
| Precision                 | 0.6   | tp/(tp+fp) = 3/(3+2) |
| Recall                    | 0.75  | tp/(tp+fn) = 3/(3+1) |
| F1                        | 0.66  | (2 * precision * recall) / (precision + recall) |
| False Positive Rate (fpr) | 0.66  | fp/(fp+tn) = 2/(2+1) |
| Accuracy                  | 0.57  | (tp + tn) / (tp + tn + fp+ fn) = (3+1)/(3+1+2+1) |

The metrics shown above have been calculated globally, across all queries. Separate versions of the metrics are calculated per query and averaged across all queries. See [metrics calculated](#metrics-calculated) below for the full list of metrics and the names used in the output.

Running the script:

```
$ python3 eval_predictions.py -g samples/ground_truth_set1.tsv -p samples/predictions_set1.tsv

{'precision': 0.6, 'recall': 0.75, 'f1': 0.6666666666666665, 'tpr': 0.75, 'fpr': 0.6666666666666666, 'accuracy': 0.5714285714285714, 'ave_precision': 0.7777777777777778, 'ave_recall': 0.8333333333333334, 'ave_f1': 0.7222222222222222, 'ave_tpr': 0.8333333333333334, 'ave_fpr': 0.6666666666666666, 'ave_accuracy': 0.611111111111111, 'l2h_ndcg10': 0.0, 'h2l_ndcg10': 0.0}
```

The price low-to-high and high-to-low NDCG scores are zero in the run above. Adding `-d samples/documents_set1.tsv` to the command will calculate NDCG scores.

## Running unit tests

Some basic unit tests are available. These are useful when modifying the main scoring function. These tests are also run when a pull request is issued. To run the tests locally:

```
$ python3 tests.py
```

## Metrics calculated

The list of calculated metrics is given below. A subset of these metrics are included on the evalAI leaderboard, but all are available in the result file available on the My Submissions page. Calculations for several of the metrics is shown earlier in this document.

* `precision` - Global precision. (Global in that individual queries are not considered.)
* `recall` - Global recall.
* `f1` - Global F1.
* `tpr` - Global true positive rate.
* `fpr` - Global false positive rate.
* `accuracy` - Global accuracy.
* `ave_precision` - Average precision. (Calculated per query and averaged across all queries.)
* `ave_recall` - Average recall.
* `ave_f1` - Average F1.
* `ave_tpr` - Average true positive rate.
* `ave_fpr` - Average false positive rate.
* `ave_accuracy` - Average accuracy.
* `l2h_ndcg10` - Price low-to-high NDCG at rank 10.
* `h2l_ndcg10` - Price high-to-low NDCG at rank 10.

Note that "average precision" metric calculated here differs from the version often used in ranked retrieval (see [average precision](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision)). There is no sorting used in these scenarios, so average precision, average recall, average F1, etc., are calculated by computing the aggregate metric for each query and taking the average of all the queries.

The price-based NDCG metrics are an early stage idea. They are calculated by grouping documents by price bucket and using the price buckets as a proxy for a relevance score.
