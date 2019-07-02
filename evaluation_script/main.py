import os.path
from . import __path__ as ROOT_PATH
from .utils import get_file_extension
from .utils import open_file
from .metrics import Metrics
from .ground_truth import GroundTruth

def calculate_query_level_metrics(query_level_metrics, ground_truth):
    """Calculates query level metrics from query level base metrics.

    Query level metrics include recall, precision, and similar metrics.
    These are calculated on a per-query basis. A set of metrics
    averaged across all queries is calculated and returned in a tuple.

    Base metrics includes counts like true positives, false positives, etc. These are
    calculated while walking the predictions.
    """
    qa_precision = 0
    qa_recall = 0
    qa_fpr = 0
    qa_accuracy = 0
    qa_f1 = 0
    qa_l2h_ndcg10 = 0
    qa_h2l_ndcg10 = 0
    total_queries = len(query_level_metrics.keys())

    for query_id in query_level_metrics:
        metrics = query_level_metrics[query_id]
        qa_precision += metrics.precision()
        qa_recall += metrics.recall()
        qa_fpr += metrics.fpr()
        qa_accuracy += metrics.accuracy()
        qa_f1 += metrics.f1()
        if ground_truth.have_document_prices:
            qa_l2h_ndcg10 += metrics.l2h_ndcg(10, ground_truth.query_truth[query_id], 5)
            qa_h2l_ndcg10 += metrics.h2l_ndcg(10, ground_truth.query_truth[query_id], 5)

    qa_precision /= total_queries
    qa_recall /= total_queries
    qa_fpr /= total_queries
    qa_accuracy /= total_queries
    qa_f1 /= total_queries
    qa_l2h_ndcg10 /= total_queries
    qa_h2l_ndcg10 /= total_queries

    return (qa_precision, qa_recall, qa_fpr, qa_accuracy, qa_f1, qa_l2h_ndcg10, qa_h2l_ndcg10)

def calculate_base_metrics(prediction_file, ground_truth):
    """Processes a prediction file and calculates base metrics.

    Each query-document pair in the prediction file is examined one-at-a-time. Each
    prediction is compared to the value in ground_truth object from the ground_truth file.

    Base metrics are counts of true positives, false positives, etc. They are computed
    both globaly and per-query. Aggregate base metrics (stored in the Metrics class object) and
    per-query base metrics (stored in the query_level_metrics dictionary) are returned in a tuple.
    """
    global_metrics = Metrics(False)
    query_level_metrics = {}
    for query_id in ground_truth.queries_with_ground_truth:
        query_level_metrics[query_id] = Metrics(True)

    # Predicted_keys serves dual purpose.
    # 1. Prevents the case where a (query_id, doc_id) pair is present multiple times.
    # 2. Helps us penalize those (query_id, doc_id) pairs that are present in ground truth
    #    but are absent in the prediction file (unlikely though)
    predicted_keys = set()
    with open_file(prediction_file) as f:
        header = f.readline().strip("\n").split("\t")
        for line in f:
            fields = line.strip("\n").split("\t")
            doc_id = fields[0]
            if doc_id in ground_truth.documents_with_ground_truth:
                for i in range(1, len(fields)):
                    query_id = header[i]
                    if (query_id, doc_id) in ground_truth.querydoc_labels:
                        if (query_id, doc_id) not in predicted_keys:
                            predicted_keys.add((query_id, doc_id))
                            truth_label = ground_truth.querydoc_labels[(query_id, doc_id)]
                            prediction_label = fields[i]
                            doc_price = ground_truth.document_price.get(doc_id, 0)
                            global_metrics.add_prediction(truth_label, prediction_label, doc_id, doc_price)
                            query_level_metrics[query_id].add_prediction(truth_label, prediction_label,
                                                                         doc_id, doc_price)

    # An unlikely case where (query_id, doc_id) pairs are present in
    # the ground truth but are absent in the prediction file
    for (query_id, doc_id) in ground_truth.querydoc_labels.keys():
        if (query_id, doc_id) not in predicted_keys:
            truth_label = ground_truth.querydoc_labels[(query_id, doc_id)]
            prediction_label = '-1' if truth_label == '1' else '1'
            doc_price = ground_truth.document_price.get(doc_id, 0)
            global_metrics.add_prediction(truth_label, prediction_label, doc_id, doc_price)
            query_level_metrics[query_id].add_prediction(truth_label, prediction_label,
                                                         doc_id, doc_price)

    return (global_metrics, query_level_metrics)

def evaluate_submission(ground_truth_file, prediction_file, doc_file=None):
    """This is contains the core functionality to evaluate a submission. When run
    in the context of the evalAI framework this function is called by the evaluate()
    function.
    """

    precision = 0
    recall = 0
    fpr = 0
    accuracy = 0
    f1 = 0
    qa_precision = 0
    qa_recall = 0
    qa_fpr = 0
    qa_accuracy = 0
    qa_f1 = 0

    ground_truth = GroundTruth(ground_truth_file, doc_file)
    # We populate query_level_base_metrics with queries with any judgements as keys.
    # This will be zero if ground truth file is all empty or no queries are judged.
    # Note that this test prevents all base metrics to be zero after calculate_base_metrics().

    if len(ground_truth.queries_with_ground_truth) > 0:
        extension = get_file_extension(prediction_file)
        if extension == "tsv" or extension == "gz":
            (global_metrics, query_level_metrics) = calculate_base_metrics(prediction_file, ground_truth)
            precision = global_metrics.precision()
            recall = global_metrics.recall()
            fpr = global_metrics.fpr()
            accuracy = global_metrics.accuracy()
            f1 = global_metrics.f1()
            (qa_precision, qa_recall, qa_fpr, qa_accuracy, qa_f1, qa_l2h_ndcg10, qa_h2l_ndcg10) = \
                calculate_query_level_metrics(query_level_metrics, ground_truth)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr": recall,
        "fpr": fpr,
        "accuracy": accuracy,
        "ave_precision": qa_precision,
        "ave_recall": qa_recall,
        "ave_f1": qa_f1,
        "ave_tpr": qa_recall,
        "ave_fpr": qa_fpr,
        "ave_accuracy": qa_accuracy,
        "l2h_ndcg10": qa_l2h_ndcg10,
        "h2l_ndcg10": qa_h2l_ndcg10
    }

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    """Evaluates the submission for a particular challenge phase and returns scores.
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            'execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """
    print("Starting Evaluation.....")

    if phase_codename == "unsupervised" or phase_codename == "supervised" or phase_codename == "final":
        print("evaluating for " +phase_codename+ " phase")

    # See if there is a documents file in the package directory
    doc_file = None
    package_dir = ROOT_PATH[0]
    doc_file_tsv_path = os.path.join(package_dir, "documents.tsv")
    doc_file_tsv_gz_path = os.path.join(package_dir, "documents.tsv.gz")
    
    if os.path.exists(doc_file_tsv_path):
        doc_file = doc_file_tsv_path
    elif os.path.exists(doc_file_tsv_gz_path):
        doc_file = doc_file_tsv_gz_path

    if doc_file is not None:
        print("Document metadata found.")

    result_values = evaluate_submission(test_annotation_file, user_submission_file, doc_file)

    if phase_codename == "unsupervised" or phase_codename == "supervised" or phase_codename == "final":
        print("completed evaluation for " +phase_codename + " phase")
    else:
        print("Invalid phase name: " +phase_codename)
        for key in result_values:
            result_values[key] = 0

    output = {}
    output["result"] = [
        {
            "data": result_values
        }
    ]
    output["submission_result"] = output["result"][0]["data"]
    return output
