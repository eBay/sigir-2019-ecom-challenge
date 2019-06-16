from .utils import get_file_extension
from .utils import open_file
from .utils import are_all_base_metrics_zero
from .utils import calculate_precision
from .utils import calculate_recall
from .utils import calculate_fpr
from .utils import calculate_f1
from .utils import BaseMetrics
from .utils import Metrics

query_level_base_metrics = {}
query_level_metrics = {}
documents_with_ground_truth = set()

"""
    This function calculates query level metrics from query level base metrics
"""
def calculate_query_level_metrics():
    qa_precision = 0
    qa_recall = 0
    qa_fpr = 0
    qa_f1 = 0

    total_queries = len(query_level_base_metrics.keys())
    if total_queries > 0:
        for query_id in query_level_base_metrics:
            base_metrics = query_level_base_metrics[query_id]
            if not are_all_base_metrics_zero(base_metrics.tp, base_metrics.tn, base_metrics.fp, base_metrics.fn):
                query_level_metrics[query_id].precision = calculate_precision(base_metrics.tp, base_metrics.fp)
                query_level_metrics[query_id].recall = calculate_recall(base_metrics.tp, base_metrics.fn)
                query_level_metrics[query_id].fpr = calculate_fpr(base_metrics.fp, base_metrics.tn)
                query_level_metrics[query_id].f1 = calculate_f1(base_metrics.tp, base_metrics.fp, base_metrics.fn)

        for query_id in query_level_base_metrics:
            qa_precision = qa_precision + query_level_metrics[query_id].precision
            qa_recall = qa_recall + query_level_metrics[query_id].recall
            qa_fpr = qa_fpr + query_level_metrics[query_id].fpr
            qa_f1 = qa_f1 + query_level_metrics[query_id].f1

        qa_precision = float(qa_precision) / total_queries
        qa_recall = float(qa_recall) / total_queries
        qa_fpr = float(qa_fpr) / total_queries
        qa_f1 = float(qa_f1) / total_queries

    return (qa_precision, qa_recall, qa_fpr, qa_f1)
"""
    This creates a map of index with header entries.
    Eg - {1:query_id_1, 2:query_id_2, 3:query_id_3}
    This breaks after reading the first line
"""
def populate_index_map(infile):
    index_map = {}
    with open_file(infile) as f:
        for line in f:
            line = line.strip("\n")
            arr = line.split("\t")
            for i in range(1, len(arr)):
                index_map[i] = arr[i]
                if arr[i] not in query_level_base_metrics:
                    query_level_base_metrics[arr[i]] = BaseMetrics()
                    query_level_metrics[arr[i]] = Metrics()
            break
    return index_map

"""
   This skips the first line (header).
   This creates a map where key is (query_id,doc_id) and
   value is the prediction (-1,0 or 1 or anything that people put but are not supposed to put)
"""
def calculate_base_metrics(infile, truth):
    """ predicted_keys serves dual purpose.
        1. Prevents the case where a (query_id, doc_id) pair is present multiple times.
        2. Helps us penalize those (query_id, doc_id) pairs that are present in ground truth
           but are absent in the prediction file (unlikely though) 
    """
    predicted_keys = set()
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    index = populate_index_map(infile)
    with open_file(infile) as f:
        next(f)
        for line in f:
            line = line.strip("\n")
            arr = line.split("\t")
            length = len(arr)
            doc_id = arr[0]
            if doc_id in documents_with_ground_truth:
                for i in range(1, length):
                    query_id = index[i]
                    if (query_id, doc_id) in truth:
                        if (query_id, doc_id) not in predicted_keys:
                            predicted_keys.add((query_id, doc_id))
                            if truth[(query_id, doc_id)] == arr[i]:
                                if arr[i] == '1':
                                    tp = tp + 1
                                    query_level_base_metrics[query_id].add_tp(1)
                                else:
                                    tn = tn + 1
                                    query_level_base_metrics[query_id].add_tn(1)
                            else:
                                if truth[(query_id, doc_id)] == '1':
                                    fn = fn + 1
                                    query_level_base_metrics[query_id].add_fn(1)
                                else:
                                    fp = fp + 1
                                    query_level_base_metrics[query_id].add_fp(1)

    """ An unlikely case where (query_id, doc_id) pairs are present in 
        the groung truth but are absent in the prediction file
    """
    for (query_id, doc_id) in truth.keys():
        if (query_id, doc_id) not in predicted_keys:
            if truth[(query_id, doc_id)] == '1':
                fn = fn + 1
                query_level_base_metrics[query_id].add_fn(1)
                """ assume that prediction is -1 """
            else:
                fp = fp + 1 
                query_level_base_metrics[query_id].add_fp(1)
                """ assume that prediction is 1 """
    return (tp, tn, fp, fn)
 
"""
   This skips the first line of a ground truth file (header).
   This creates a map where key is (query_id,doc_id) and
   value is the prediction (-1,0 or 1 or anything that people put but are not supposed to put)
"""
def populate_ground_truth(infile):
    index = populate_index_map(infile)
    results = {}
    with open_file(infile) as f:
        next(f)
        for line in f:
            line = line.strip("\n")
            arr = line.split("\t")
            length = len(arr)
            doc_id = arr[0]
            for i in range(1, length):
                if arr[i] != '0':
                    documents_with_ground_truth.add(doc_id)
                    query_id = index[i]
                    results[(query_id, doc_id)] = arr[i]
    return results

def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    """
    Evaluates the submission for a particular challenge phase adn returns score
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
            execution_time': u'123',
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
    query_level_base_metrics.clear()
    query_level_metrics.clear()
    documents_with_ground_truth.clear()
    truth = populate_ground_truth(test_annotation_file)
    output = {}
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    precision = 0
    recall = 0
    fpr = 0
    f1 = 0
    qa_precision = 0
    qa_recall = 0
    qa_fpr = 0
    qa_f1 = 0

    if phase_codename == "unsupervised" or phase_codename == "supervised" or phase_codename == "final":
        print("evaluating for " +phase_codename+ " phase")
        extension = get_file_extension(user_submission_file)
        if extension == "tsv" or extension == "gz": 
            (tp, tn, fp, fn) = calculate_base_metrics(user_submission_file, truth)
            if not are_all_base_metrics_zero(tp, tn, fp, fn):
                precision = calculate_precision(tp, fp)
                recall = calculate_recall(tp, fn)
                fpr = calculate_recall(fp, tn)
                f1 = calculate_f1(tp, fp, fn)
                (qa_precision, qa_recall, qa_fpr, qa_f1) = calculate_query_level_metrics()
 
        output["result"] = [
            {
                "data": {
                    "global_precision": precision,
                    "global_recall": recall,
                    "global_f1": f1,
                    "global_tpr": recall,
                    "global_fpr": fpr,
                    "average_precision": qa_precision,
                    "average_recall": qa_recall,
                    "average_f1": qa_f1,
                    "average_tpr": qa_recall,
                    "average_fpr": qa_fpr
                }
            }
        ]
        output["submission_result"] = output["result"][0]["data"]
        print("completed evaluation for " +phase_codename + " phase")

    return output
