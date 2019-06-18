import os
import gzip
def get_file_extension(infile):
    filename = os.path.split(infile)[1]
    filename_arr = filename.split(".")
    return filename_arr[len(filename_arr)-1]

def open_file(filename):
    f = open(filename, 'rb')
    is_gzipped = f.read(2) == b'\x1f\x8b'
    f.close()
    r = gzip.open(filename, 'rt') if is_gzipped else open(filename, 'rt')
    return r

def calculate_precision(tp, fp):
    """Return the precision given the number of true positives (tp) and false positives (fp).
    Precision is 1 when fp is zero as there are no spurious results, even if tp is zero.
    """
    return float(tp) / (tp + fp) if fp != 0 else 1

def calculate_recall(tp, fn):
    """Return the recall given the number of true positives (tp) and false negatives (fn).
    Recall is 1 when fn is zero as there are no spurious results, even if tp is zero.
    """
    return float(tp) / (tp + fn) if fn != 0 else 1

def calculate_tpr(tp, fn):
    """Return the true positive rate given the number of true positives (tp) and false
    negatives (fp). True positive rate is 1 when fn is zero as there are no spurious
    results, even if tp is zero.
    """
    return float(tp) / (tp + fn) if fn != 0 else 1

def calculate_fpr(fp, tn):
    """Return the false positive rate given the number of false positives (fp) and true
    negatives (tn). False positive rate is 1 when tn is zero as there are no spurious
    results, even if fp is zero.
    """
    return float(fp) / (fp + tn) if tn != 0 else 1

def calculate_accuracy(tp, fp, tn, fn):
    """Returns the 'accuracy', defined as the number of correct predictions divided by the
    total predictions.
    """
    total_predictions = tp + fp + tn + fn;
    return float(tp + tn) / total_predictions if total_predictions != 0 else 1

def calculate_f1(tp, fp, fn):
    f1 = 0
    if tp == 0 and fp == 0 and fn == 0:
        return f1
    else:
        p = calculate_precision(tp,fp)
        r = calculate_recall(tp,fn) 
        if p > 0 or r > 0:
            f1 = 2 * (p * r) / (p + r)
        return f1

class BaseMetrics():
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def add_tp(self, x):
        self.tp = self.tp + x

    def add_fp(self, x):
        self.fp = self.fp + x

    def add_tn(self, x):
        self.tn = self.tn + x

    def add_fn(self, x):
        self.fn = self.fn + x

class Metrics():
    def __init__(self):
        self.precision = 0
        self.recall = 0
        self.fpr = 0
        self.accuracy = 0
        self.f1 = 0
