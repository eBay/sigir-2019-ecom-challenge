import os
import gzip
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
            break
    return index_map

def get_file_extension(infile):
    filename = os.path.split(infile)[1]
    filename_arr = filename.split(".")
    return filename_arr[len(filename_arr)-1]

def open_file(filename):
    f = open(filename,'rb')
    is_gzipped = f.read(2) == b'\x1f\x8b'
    f.close()
    r = gzip.open(filename,'rt') if is_gzipped else open(filename, 'rt')
    return r

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
            for i in range(1, length):
                query_id = index[i]
                if (query_id, doc_id) in truth:
                    if (query_id, doc_id) not in predicted_keys:
                        predicted_keys.add((query_id, doc_id))
                        if truth[(query_id, doc_id)] == arr[i]:
                            if arr[i] == '1':
                                tp = tp + 1
                            else:
                                tn = tn + 1
                        else:
                            if truth[(query_id, doc_id)] == '1':
                                fn = fn + 1
                            else:
                                fp = fp + 1

    """ An unlikely case where (query_id, doc_id) pairs are present in 
        the groung truth but are absent in the prediction file
    """
    for key in truth.keys():
        if key not in predicted_keys:
            if truth[key] == '1':
                fn = fn + 1 
                """ assume that prediction is -1 """
            else:
                fp = fp + 1 
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
                    query_id = index[i]
                    results[(query_id, doc_id)] = arr[i]
    return results

def calculate_precision(tp, fp):
    precision = 0
    if tp == 0 and fp == 0:
        """Precision = 1 when FP=0, since no there were no spurious results"""
        precision = 1
    else:
        precision = float(tp) / (tp + fp)
    return precision

def calculate_recall(tp, fn):
    recall = 0
    if tp == 0 and fn == 0:
        """Recall = 1 when FN=0, since 100% of the TP were discovered""" 
        recall = 1
    else:
        recall = float(tp) / (tp + fn)
    return recall

def calculate_tpr(tp, fn):
    return calculate_recall(tp, fn)

def calculate_fpr(fp, tn):
    fpr = 0
    if fp == 0 and tn == 0:
        fpr = 1
    else:
        fpr = float(fp) / (fp + tn)
    return fpr

def calculate_f1(precision, recall):
    f1 = 0
    if precision > 0 or recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1

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
    if phase_codename == "unsupervised" or phase_codename == "supervised" or phase_codename == "final":
        print("evaluating for " +phase_codename+ " phase")
        extension = get_file_extension(user_submission_file)
        if extension == "tsv" or extension == "gz": 
            (tp, tn, fp, fn) = calculate_base_metrics(user_submission_file, truth)
            precision = calculate_precision(tp, fp)
            recall = calculate_recall(tp, fn)
            fpr = calculate_recall(fp, tn)
            f1 = calculate_f1(precision, recall)
        else:
            precision = 0
            recall = 0
            fpr = 0
            f1 = 0
        output["result"] = [
            {
                "data": {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "tpr": recall,
                    "fpr": fpr,
                }
            }
        ]
        output["submission_result"] = output["result"][0]["data"]
        print("completed evaluation for " +phase_codename + " phase")

    return output
