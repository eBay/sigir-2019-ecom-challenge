import math
from .ground_truth import Document
from .ground_truth import QueryTruth
from .ground_truth import GroundTruth
from .binutils import ExpBins

class DocumentPrediction():
    """Holds the doc_id, price, and ground truth label of a document predicted as relevant."""
    def __init__(self, doc_id, truth_label, price):
        self.doc_id = doc_id
        self.truth_label = truth_label
        self.price = price

class QueryPrediction():
    """Holds the prediction data for an individual query.
    
    This class holds data needed for NDCG style price sort calculations. It keeps the
    documents predicted to be relevant. This set is limited to the subset of query-document
    pairs having judgements.
    """
    def __init__(self):
        self.documents_predicted_relevant = []

    @property
    def num_predicted_relevant(self):
        return len(self.documents_predicted_relevant)

    def add_doc_predicted_relevant(self, doc_id, truth_label, price=0.0):
        self.documents_predicted_relevant.append(DocumentPrediction(doc_id, truth_label, price))

    def sort_docs_by_price(self, reverse=False):
        """Sort the documents by price. By default sort is low-to-high."""
        self.documents_predicted_relevant.sort(key=lambda doc: doc.price, reverse=reverse)
        

class Metrics():
    def __init__(self, is_query_level=True):
        self.is_query_level = is_query_level
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.query_prediction = QueryPrediction()

    def add_prediction(self, truth_label, prediction, doc_id, doc_price=0):
        """Adds a prediction. The truth label should be 1 or -1 from the ground truth
        file. The prediction label should be 1 or -1 from the prediction file. The
        prediction is assumed to be incorrect if it is not a 1 or -1 value.
        """
        assert (truth_label == '1' or truth_label == '-1')
        
        if truth_label == prediction:
            if truth_label == '1':
                self.tp += 1
            else:
                self.tn += 1
        else:
            if truth_label == '1':
                self.fn += 1
            else:
                self.fp += 1

        if self.is_query_level and prediction != '-1':
            self.query_prediction.add_doc_predicted_relevant(doc_id, truth_label, doc_price)

    def precision(self):
        """Return the precision given the number of true positives (tp) and false positives (fp).
        Precision is 1 when fp is zero as there are no spurious results, even if tp is zero.
        """
        return float(self.tp) / (self.tp + self.fp) if self.fp != 0 else 1
    
    def recall(self):
        """Return the recall given the number of true positives (tp) and false negatives (fn).
        Recall is 1 when fn is zero as there are no spurious results, even if tp is zero.
        """
        return float(self.tp) / (self.tp + self.fn) if self.fn != 0 else 1
    
    def fpr(self):
        """Return the false positive rate given the number of false positives (fp) and true
        negatives (tn). False positive rate is 1 when tn is zero as there are no spurious
        results, even if fp is zero.
        """
        return float(self.fp) / (self.fp + self.tn) if self.tn != 0 else 1
    
    def accuracy(self):
        """Returns the 'accuracy', defined as the number of correct predictions divided by the
        total predictions.
        """
        total_predictions = self.tp + self.fp + self.tn + self.fn;
        return float(self.tp + self.tn) / total_predictions if total_predictions != 0 else 1
    
    def f1(self):
        f1 = 0
        if self.tp == 0 and self.fp == 0 and self.fn == 0:
            return f1
        else:
            p = float(self.tp) / (self.tp + self.fp) if self.fp != 0 else 1
            r = float(self.tp) / (self.tp + self.fn) if self.fn != 0 else 1
            if p > 0 or r > 0:
                f1 = 2 * (p * r) / (p + r)
            return f1

    def l2h_ndcg(self, n, query_truth, num_bins=5):
        """A simple price low-to-high NDCG calculation. This calculate divides the price
        range into exponentially sized buckets and assigns each item a score based on the
        bucket it falls into. This score is used with a traditional log-based discounting
        by rank position to construct an NDCG score.
        """
        if not self.is_query_level:
            return 0.0
        if query_truth.num_relevant == 0:
            return 1.0

        query_truth.sort_docs_by_price()
        self.query_prediction.sort_docs_by_price()
        
        lowest_price = query_truth.relevant_documents[0].price
        highest_price = query_truth.relevant_documents[query_truth.num_relevant - 1].price
        if (lowest_price == highest_price):
            highest_price += 1
        expbins = ExpBins(lowest_price, highest_price, num_bins, exp_base=math.exp(1))
        
        idcg = 0
        for i in range(min(n, query_truth.num_relevant)):
            doc = query_truth.relevant_documents[i]
            doc_price = doc.price
            relevance_score = num_bins + 1 - expbins.getbin(doc_price)
            idcg += relevance_score / math.log(i + 2, 2)

        dcg = 0
        for i in range(min(n, self.query_prediction.num_predicted_relevant)):
            doc = self.query_prediction.documents_predicted_relevant[i]
            doc_price = doc.price
            if doc.truth_label == '1':
                relevance_score = num_bins + 1 - expbins.getbin(doc_price)
                dcg += relevance_score / math.log(i + 2, 2)

        ndcg = dcg / idcg
        # print("L2H NDCG@", n, "- relevant:", query_truth.num_relevant,
        #      "predicted:", self.query_prediction.num_predicted_relevant,
        #      "price range:", lowest_price, "-", highest_price, 
        #      "idcg:", idcg, "dcg: ", dcg, "ndcg: ", ndcg) 

        return ndcg

    def h2l_ndcg(self, n, query_truth, num_bins=5):
        """A simple price high-to-low NDCG calculation. This calculate divides the price
        range into exponentially sized buckets and assigns each item a score based on the
        bucket it falls into. This score is used with a traditional log-based discounting
        by rank position to construct an NDCG score.
        """
        if not self.is_query_level:
            return 0.0
        if query_truth.num_relevant == 0:
            return 1.0

        query_truth.sort_docs_by_price(reverse=True)
        self.query_prediction.sort_docs_by_price(reverse=True)
        
        highest_price = query_truth.relevant_documents[0].price
        lowest_price = query_truth.relevant_documents[query_truth.num_relevant - 1].price
        if (lowest_price == highest_price):
            highest_price += 1
        expbins = ExpBins(lowest_price, highest_price, num_bins, exp_base=math.exp(1), invert=True)
        
        idcg = 0
        for i in range(min(n, query_truth.num_relevant)):
            doc = query_truth.relevant_documents[i]
            doc_price = doc.price
            relevance_score = expbins.getbin(doc_price) + 1
            idcg += relevance_score / math.log(i + 2, 2)

        dcg = 0
        for i in range(min(n, self.query_prediction.num_predicted_relevant)):
            doc = self.query_prediction.documents_predicted_relevant[i]
            doc_price = doc.price
            if doc.truth_label == '1':
                relevance_score = expbins.getbin(doc_price) + 1
                dcg += relevance_score / math.log(i + 2, 2)

        ndcg = dcg / idcg
        # print("H2L NDCG@", n, "- relevant:", query_truth.num_relevant,
        #       "predicted:", self.query_prediction.num_predicted_relevant,
        #       "price range:", lowest_price, "-", highest_price,
        #       "bin size:", expbins.bin_size, "breaks:", expbins.breaks, 
        #       "idcg:", idcg, "dcg: ", dcg, "ndcg: ", ndcg) 

        return ndcg
