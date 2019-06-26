from .utils import open_file

class Document():
    """Holds the doc_id and price of a document."""
    def __init__(self, doc_id, price=0.0):
        self.doc_id = doc_id
        self.price = price

class QueryTruth():
    """Holds the ground truth data for an individual query.

    This class holds data needed for NDCG style price sort calculations. It keeps the
    relevant documents for the query as well as the document price.
    """
    def __init__(self):
        self.num_judgements = 0;
        self.relevant_documents = []

    @property
    def num_relevant(self):
        return len(self.relevant_documents)

    def add_judgement(self, label, doc_id):
        assert label == '1' or label == '-1'
        self.num_judgements += 1
        if label == '1':
            self.relevant_documents.append(Document(doc_id))

    def update_doc_prices(self, document_prices):
        """"Updates all the prices of a query's relevant docs."""
        for doc in self.relevant_documents:
            doc.price = document_prices[doc.doc_id]

    def sort_docs_by_price(self, reverse=False):
        """Sort the documents by price. By default sort is low-to-high."""
        self.relevant_documents.sort(key=lambda doc: doc.price, reverse=reverse)
            
class GroundTruth():
    """GroundTruth holds the state of a 'ground truth' file.

    The GroundTruth structure is constructed for fast processing of a prediction
    file. A set of documents with judgements and a map from (query_id, doc_id) pairs
    to judgement labels is used for basic scoring (recall, precision, f1). DCG style
    scoring is supporting by keeping a list of all relevant documents associated
    with a query and a map of prices for documents having a relevant judgement.

    Args:
        ground_truth_file: File with ground truth data.
        document_file (optional): File with document data used to record prices.

    Attributes:
        documents_with_ground_truth: A set of doc_ids having at least one judgement.
        querydoc_labels: A dictionary mapping (query_id, doc_id) pairs to judgements.
        query_truth: A dictionary mapping query_ids to QueryTruth instances.
        relevant_doc_queries: A dictionary mapping doc_ids to a list of relevant queries.
        queries_with_ground_truth: A set of query_ids having at least one judgement.
        document_price: A dictionary mapping doc_ids to price. Includes documents having
            one or more relevance labels. Populated if a document_file is provided.
        have_document_prices: Boolean indicating if document prices are included.
    """
    
    def __init__(self, ground_truth_file, document_file=None):
        self.documents_with_ground_truth = set()
        self.querydoc_labels = {}
        self.query_truth = {}
        self.relevant_doc_queries = {}
        self.document_price = {}
        self.have_document_prices = False
        
        self._read_ground_truth(ground_truth_file)
        if document_file is not None:
            self.have_document_prices = True
            self._read_doc_prices(document_file)
            self._update_queries_with_prices()
            # self._print_queries()

    @property
    def queries_with_ground_truth(self):
        return self.query_truth.keys()

    def _read_ground_truth(self, ground_truth_file):
        """Reads a ground truth file and populates the class instance.
        
        A ground truth file has a matrix of relevance labels for (doc_id, query_id)
        pairs. The first line is a header. The first column contains doc_ids,
        subsequent columns contain a label for each query. The query_id is found in
        the header line. Fields are tab separated. Relevance labels are members of 
        {'-1', '1', '0' }, meaning not-relevant, relevant, not-judged. An example
        of a ground truth file with three documents and three queries:
            doc/query   1     2     3
            1001        1     0     1
            1002       -1     0     0
            1003        0     1    -1 
        """
        with open_file(ground_truth_file) as f:
            header = f.readline().strip("\n").split("\t")
            for line in f:
                fields = line.strip("\n").split("\t")
                doc_id = fields[0]
                for i in range(1, len(fields)):
                    label = fields[i]
                    if label != '0':
                        query_id = header[i]
                        self.documents_with_ground_truth.add(doc_id)
                        self.querydoc_labels[(query_id, doc_id)] = label
                        if query_id not in self.query_truth:
                            self.query_truth[query_id] = QueryTruth()
                        self.query_truth[query_id].add_judgement(label, doc_id)
                        if label == '1':
                            if doc_id not in self.relevant_doc_queries:
                                self.relevant_doc_queries[doc_id] = []
                            self.relevant_doc_queries[doc_id].append(query_id)

    def _read_doc_prices(self, document_file):
        """Reads an auxilliary metadata containing price data for documents. This function
        considers only two fields, 'doc_id' and 'price', which it locates from the header
        row. No other fields are needed and are ignored if found.
        """
        with open_file(document_file) as f:
            header = f.readline().strip("\n").split("\t")
            doc_id_index = header.index('doc_id')  # Raises an error if not found
            price_index = header.index('price')
            for line in f:
                fields = line.strip("\n").split("\t")
                doc_id = fields[doc_id_index]
                if doc_id in self.documents_with_ground_truth:
                    self.document_price[doc_id] = float(fields[price_index])

    def _update_queries_with_prices(self):
        for q in self.query_truth.values():
            q.update_doc_prices(self.document_price)

    def _print_queries(self):
        """Debugging routine."""
        for (qid, qtruth) in self.query_truth.items():
            print("query_id: ", qid)
            print("   num_judgments: ", qtruth.num_judgements, "; num_relevant: ", len(qtruth.relevant_documents))
            print("   relevant_docs:")
            for d in qtruth.relevant_documents:
                print("      ", d.doc_id, ", price: ", d.price)
            
