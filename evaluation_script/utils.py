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

class Metrics():
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

    def calculate_precision(self):
        """Return the precision given the number of true positives (tp) and false positives (fp).
        Precision is 1 when fp is zero as there are no spurious results, even if tp is zero.
        """
        return float(self.tp) / (self.tp + self.fp) if self.fp != 0 else 1
    
    def calculate_recall(self):
        """Return the recall given the number of true positives (tp) and false negatives (fn).
        Recall is 1 when fn is zero as there are no spurious results, even if tp is zero.
        """
        return float(self.tp) / (self.tp + self.fn) if self.fn != 0 else 1
    
    def calculate_fpr(self):
        """Return the false positive rate given the number of false positives (fp) and true
        negatives (tn). False positive rate is 1 when tn is zero as there are no spurious
        results, even if fp is zero.
        """
        return float(self.fp) / (self.fp + self.tn) if self.tn != 0 else 1
    
    def calculate_accuracy(self):
        """Returns the 'accuracy', defined as the number of correct predictions divided by the
        total predictions.
        """
        total_predictions = self.tp + self.fp + self.tn + self.fn;
        return float(self.tp + self.tn) / total_predictions if total_predictions != 0 else 1
    
    def calculate_f1(self):
        f1 = 0
        if self.tp == 0 and self.fp == 0 and self.fn == 0:
            return f1
        else:
            p = float(self.tp) / (self.tp + self.fp) if self.fp != 0 else 1
            r = float(self.tp) / (self.tp + self.fn) if self.fn != 0 else 1
            if p > 0 or r > 0:
                f1 = 2 * (p * r) / (p + r)
            return f1

