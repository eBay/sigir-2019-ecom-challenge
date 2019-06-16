import os
import gzip
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

def are_all_base_metrics_zero(tp, tn, fp, fn):
    if tp == 0 and tn == 0 and fp == 0 and fn == 0:
        return True
    else:
        return False

def calculate_precision(tp, fp):
    if tp == 0 and fp == 0:
        """Precision = 1 when FP=0, since no there were no spurious results"""
        return 1
    else:
        return float(tp) / (tp + fp)

def calculate_recall(tp, fn):
    if tp == 0 and fn == 0:
        """Recall = 1 when FN=0, since 100% of the TP were discovered""" 
        return 1
    else:
        return float(tp) / (tp + fn)

def calculate_tpr(tp, fn):
    if tp == 0 and fn == 0:
        """TPR = 1 when FN=0, since 100% of the TP were discovered""" 
        return 1
    else:
        return float(tp) / (tp + fn)

def calculate_fpr(fp, tn):
    if fp == 0 and tn == 0:
        return 1
    else:
        return float(fp) / (fp + tn)

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
        self.f1 = 0
