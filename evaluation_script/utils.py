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
    tpr = 0
    if tp == 0 and fn == 0:
        """TPR = 1 when FN=0, since 100% of the TP were discovered""" 
        tpr = 1
    else:
        tpr = float(tp) / (tp + fn)
    return tpr

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

class BaseMetrics():
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def add_tp(self, x):
        self.tp = self.tp + x

    def return_tp(self):
        return self.tp

    def add_fp(self, x):
        self.fp = self.fp + x

    def return_fp(self):
        return self.fp

    def add_tn(self, x):
        self.tn = self.tn + x

    def return_tn(self):
        return self.tn

    def add_fn(self, x):
        self.fn = self.fn + x

    def return_fn(self):
        return self.fn

class Metrics():
    def __init__(self):
        self.precision = 0
        self.recall = 0
        self.fpr = 0
        self.f1 = 0
