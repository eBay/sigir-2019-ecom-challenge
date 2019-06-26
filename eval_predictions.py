import evaluation_script
import argparse

parser = argparse.ArgumentParser(description='Evaluation script used in the eBay SIGIR 2019 eCommerce Search Challenge.')
parser.add_argument('-g', '--ground-truth-file', required=True, help="Ground truth file")
parser.add_argument('-p', '--prediction-file', required=True, help="Prediction file")
parser.add_argument('-d', '--document-file', required=False, default=None, help="Document file")
args = parser.parse_args()

r = evaluation_script.evaluate_submission(args.ground_truth_file, args.prediction_file, args.document_file)
print();
print(r)
