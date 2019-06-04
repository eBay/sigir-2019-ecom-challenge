# eBay SIGIR 2019 eCommerce Search Challenge

This repository contains auxiliary tools used to support the [eBay SIGIR 2019 eCommerce Search Challenge: High Accuracy Recall Task](https://sigir-ecom.github.io/data-task.html). Currently this repository consists of one tool: An evaluation script challenge participants can use to score their predictions locally.

More information about the challenge can be found on the [SIGIR eCom'19 Data Challenge](https://sigir-ecom.github.io/data-task.html) page.

## Evaluation Script (eval_predictions.py)

The script provided here is a command line version of the evaluation script used on the [EvalAI](https://sigir-ecom.github.io/data-task.html) challenge hosting site to score participant submissions. The script allows participants to run the scoring function on their own train and test data sets. Other than having a command line interface it is the same as the run on the evalAI server.

The script takes two files, a "ground truth" file and a "prediction" file. These are tab-separated-value files in the same format used for challenge submissions. An example run:

```
$ python3 eval_predictions.py --ground-truth-file ground_truth.tsv  --prediction-file predictions.tsv
Starting Evaluation.....
evaluating for supervised phase
completed evaluation for supervised phase

{'precision': 0.7703079281036644, 'recall': 0.7658528138528139, 'f1': 0.7680739107028117, 'tpr': 0.7658528138528139, 'fpr': 0.1765509116709952}
```

The final output line has the same result values as used on the [challenge leaderboard](https://evalai.cloudcv.org/web/challenges/challenge-page/361/leaderboard). (The output references to the "supervised phase" can be ignored.)

The script has been tested with both python 2 and python 3. The short form options `-g` and `-p` can be used in place of the longer `--ground-truth-file` and `--prediction-file` form options. Both plain text and gzipped files are supported. (Please use `.tsv` or `.gz` file extensions.) An example call with short options and gzipped files:

```
$ python3 eval_predictions.py -g ground_truth.tsv.gz -p predictions.tsv.gz
```

An example showing the file format is given below. Challenge participants will find more information in the `readme` file distributed with the data files.

### Evaluation script example

The example uses the files in `samples/` directory show in the tables below.

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

The `ground_truth.tsv` file has labels for seven of the twelve query-document pairs (those with `1` or `-1` entries). The `predictions.tsv` file has predictions for all twelve of the query-document pairs. The `eval_predictions.py` script will generate a score based on the seven query-document pairs having relevant-or-not labels. For this pair of files:

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

Running the script:

```
$ python3 eval_predictions.py -g samples/ground_truth_set1.tsv -p samples/predictions_set1.tsv
Starting Evaluation.....
evaluating for supervised phase
completed evaluation for supervised phase

{'precision': 0.6, 'recall': 0.75, 'f1': 0.6666666666666665, 'tpr': 0.75, 'fpr': 0.6666666666666666}
```
