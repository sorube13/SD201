'''
INF-230 Tutorial on Clustering
Mario Sozio, Oana Balalau, Luis Galarraga

Simple program to test K-Means and Agglomerative Clustering on
a documents corpus.

'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.cluster import AgglomerativeClustering
from optparse import OptionParser
from sklearn.cluster import KMeans
from corpus import WikiCorpus
from corpus import Document
from scipy.spatial import distance
from collections import namedtuple
from time import time
from math import log
import sys
import operator
import mathplotlib.pyplot as plt

## Supported clustering methods
supported_algorithms = ['k-means', 'k-means++', 'agglomerative']
supported_labels = ['topic', 'type']
linkage_criteria = ['average', 'complete', 'ward']
corpus_specific_stopwords = ['january', 'february', 'march', 'april',
                             'may', 'june', 'july', 'august', 'september', 'october',
                             'november', 'december']



'''
It generates a document-term matrix from a given corpus (an object of class WikiCorpus).
The entry in position [i, j] contains the tf-idf score of the j-th term in the i-th document.
This method returns a sparse.crc_matrix (sparse matrix), the ground truth clustering as
an array of pairs (doc-title, doc-topic) sorted in descending order by doc-length (number of words)
and the vectorizer object.
'''
def tfidf_vectorize(corpus) :
    ## This object does all the job. For more information about
    ## the semantics of the arguments, please read the documentation at
    ## http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    ## TfidfVectorizer takes care of the stop-words.
    corpus_specific_stopwords.extend(ENGLISH_STOP_WORDS)
    vectorizer = TfidfVectorizer(max_df = 0.5,
                                 token_pattern='[A-Za-z]{3,}', # we restrict to words with 3 or more characters
                                 min_df = 50, # words occurring less than this number of times are discarded
                                 stop_words= corpus_specific_stopwords, # We use a standard set of stop words for the English language plus some words we already identified
                                 use_idf = True # Use tf/idf, clusterIdx.e., term frequency divided by the term's document frequency
                                 )
    ## We store the documents and the text of the articles in different lists.
    texts = []
    documents = []

    ## Iterating the corpus
    for doc in corpus :
        texts.append(doc.text)
        ## We store the title and the category as a pair
        ## The category is the topic of the article. We will use
        ## it as ground truth when calculating purity and entropy
        documents.append(doc)

    ## This call constructs the document-term matrix. It returns a sparse matrix of
    ## type http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html
    doc_term_matrix = vectorizer.fit_transform(texts);
    ## We return the ground truth clustering, the document-term matrix and the vectorizer object
    return documents, doc_term_matrix, vectorizer


def agglomerative(doc_term_matrix, k, linkage) :
    ## Documentation here:
    ## http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
    agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    print("Clustering sparse data with %s" % agg)
    t0 = time()
    ## This call does the job but it requires a dense doc_term_matrix.
    agg.fit(doc_term_matrix.todense())
    print("done in %0.3fs" % (time() - t0))
    return agg;

def kmeans(doc_term_matrix, k, centroids='random', max_iterations=300) :
    ## Documentation here:
    ## http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans
    km = KMeans(n_clusters=k, init=centroids, max_iter=max_iterations, n_init=10)
    print("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(doc_term_matrix)
    print("done in %0.3fs" % (time() - t0))
    return km

'''
It computes the sum of square error of a clustering obtained with K-Means or K-Means++.
'''
def sse(clustering, doc_term_matrix) :
    ## clustering.cluster_centers_ is a numpy.array of size k (the number of clusters)
    ## containing the centroids of each cluster. The element in position clusterIdx is the
    ## centroid of the clusterIdx-th cluster.
    ## See https://scipy-lectures.github.io/intro/numpy/array_object.html
    centroids = clustering.cluster_centers_
    totalSSE = 0.0
    ## clustering._labels_ is an array of size N = number of documents
    ## that stores the clusters labels of the documents assigned by the clustering
    ## algorithm. The labels are numbers between 0 and k. If clustering.labels_[clusterIdx] stores
    ## the value j (0 <= j < k), it means the clusterIdx-th document was assigned to the j-th
    ## cluster.
    for i in range(len(clustering.labels_)) :
        ## Obtain the clusterIdx-th row of the matrix, that is, the vector
        ## of the clusterIdx-th document
        rowi = doc_term_matrix.getrow(i).toarray()
        centroidi = centroids[clustering.labels_[i]]
        ## Calculate the distance between the document vector and the centroid
        ## of the cluster it was assigned to.
        dist = distance.euclidean(rowi, centroidi)
        totalSSE += pow(dist, 2)
    return totalSSE

'''
Builds a dictionary of the form {cluster_index: [doc_idx1, doc_idx2, ... ]}
'''
def clusterIdx2DocIdx(clustering) :
    labels2docs = {}
    n_documents = len(clustering.labels_)
    for i in range(n_documents) :
        ## Get the cluster of the clusterIdx-th document
        labeli = clustering.labels_[i]
        if labeli not in labels2docs :
            labels2docs[labeli] = [] ##Initialize the list

        labels2docs[labeli].append(i)

    return labels2docs

'''
It computes the purity of a clustering with respect to a ground truth
clustering. Refer to
http://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
to calculate the purity of a cluster given a ground truth.
The ground truth is given as an array of tuples (article-title, article-topic)
'''
def purity(clustering, ground_truth) :
    ## In this phase we are building a dictionary (hash table) of the form
    ## {cluster_label -> [document_indexes]} for the resulting clustering.
    ## For instance if we have 4 documents and 2 clusters
    ## this dictionary would look like:
    ## 0 : [0, 2]
    ## 1 : [1, 3]
    ## assuming docs 0, 2 are in the first cluster and docs 1, 3 are in the second.
    ##
    labels2docIds = clusterIdx2DocIdx(clustering)
    n_documents = len(clustering.labels_)

    ## We are ready to implement the sum of the purity
    purity = 0.0
    totalSum = 0.0
    histogram = {}
    for cluster_index in labels2docIds :
        ## TODO: You need to find the frequency of the most common ground truth
        ## label in the documents of the cluster.
        ## We recommend you to build a histogram category: # of docs, from the documents
        ## of the cluster, e.g.,
        ## {sports : 3, music : 5, politics : 4, ...}
        ## Pick the category with the biggest count
        if cluster_index not in histogram :
            histogram[cluster_index] = [] ##Initialize the list
        histogram[cluster_index] = (len(labels2docsIds[cluster_index]))
        totalSum += len(labels2docsIds[cluster_index])
    max_cluster = max(histogram, key=histogram.get)

    purity = histogram[max_cluster]/totalSum
    ## return purity
    return 0.0

'''
It computes the entropy of a clustering with respect to a ground truth
clustering.
The ground truth is given as an array of objects of the class Document. The
property category holds the category the document belongs to.
'''
def entropy(clustering, ground_truth) :
    ## In this phase we are building a dictionary (hash table) of the form
    ## {cluster_label -> [document_indexes]} for the resulting clustering.
    ## For instance if we have 4 documents and 2 clusters
    ## this dictionary would look like:
    ## 0 : [0, 2]
    ## 1 : [1, 3]
    ## assuming docs 0, 2 are in the first cluster and docs 1, 3 are in the second.
    ##
    labels2docs = clusterIdx2DocIdx(clustering)

    ## For each cluster we have the indexes of the documents
    ## that belong to that cluster. To compute the purity we need
    ## to count the most common cluster_index that appears from the gold
    ## standard
    totalSum = 0.0
    histogram_ground_truth = {}
    for cluster_index in labels2docs :
        ## TODO: You need to find the frequencies of ALL ground truth labels in the
        ## cluster to calculate the term p_{wc} * log(p_{wc}) in the formula (look at the
        ## exercise description)
        pi = 0.0
        if cluster_index not in histogram_ground_truth :
            histogram_ground_truth[cluster_index] = [] ##Initialize the list
            pi = len(labels2docsIds[cluster_index])
        histogram_ground_truth[cluster_index] = pi
        totalSum += pi * log(pi)

        ## histogram_ground_truth = {}

    ## return totalSum
    return 0.0

'''
It extracts all the truth categories associated to the input documents.
They categories are returned as set
'''
def get_categories(documents) :
    categories = set()
    for doc in documents :
        if len(doc.category) > 0 :
            categories.add(doc.category)

    return categories

'''
Output the results of the clustering in a human readable fashion to an output buffer.
'''
def output (clustering, documents, output) :
    # Dictionary of the form {clusterIdx : [doc1, ... docN]}
    labels2docs = clusterIdx2DocIdx(clustering)
    with open(output, 'wb') as fout :
        for clusterIdx in labels2docs :
            fout.write('Cluster ' + str(clusterIdx) + '\n')
            doc_indexes = labels2docs[clusterIdx]
            doc_titles = []
            for index in doc_indexes :
                if documents[index].title != '' :
                    doc_titles.append(documents[index].title + '(' + documents[index].category + ')')
                else :
                    doc_titles.append(documents[index].title)
            fout.write(', '.join(doc_titles) + '\n')


def createTestClustering() :
    TestClustering = namedtuple('TestClustering', ['labels_'])
    testTuple = TestClustering(labels_ = [0, 1, 0, 1, 0, 1, 0, 1])

    groundTruth = [Document('doc1', '', 'sports'), Document('doc2', '', 'music'),
                   Document('doc3', '', 'music'), Document('doc4', '', 'sports'),
                   Document('doc5', '', 'sports'), Document('doc6', '', 'music'),
                   Document('doc7', '', 'music'), Document('doc8', '', 'sports')]
    return testTuple, groundTruth

def testPurity() :
    ## In this routine, we will create a fake clustering
    testClustering, groundTruth = createTestClustering()
    purityValue = purity(testClustering, groundTruth)
    return purityValue == 0.5

def testEntropy() :
    testClustering, groundTruth = createTestClustering()
    entropyValue = entropy(testClustering, groundTruth)
    return entropyValue == 2

def plotSSEvsK(doc_term_matrix):
    sseArray = []
    k = list(xrange(3,16))
    for i in k:
        clustering = kmeans(doc_term_matrix, k = i, centroids = 'random')
        sseScore = sse(clustering, doc_term_matrix)
        sseArray.append(sseScore)
    plt(k, sseArray)
    plt.show()
    return


## Main program
if __name__ == '__main__' :
    # parse commandline arguments
    op = OptionParser()
    op.add_option("--algorithm", action="store",
                  dest="algorithm",
                  default="k-means",
                  help="Define the algorithm used for clustering: k-means, agglomerative, k-means++")
    op.add_option("--output", action="store",
                  dest="output", default="clusters",
                  help='''The output file where the clusters of documents will be written. The program
                  outputs only the titles of the documents. If not specified, the output is written
                  in a text file named "clusters" located in the current directory of the command line.''')
    op.add_option("--linkage", action="store",
                  dest="linkage", default="ward",
                  help='''Applicable only for agglomerative clustering: ward, complete, average.
                  It determines the linkage criterion used to compute the distance between two clusters.''')
    op.add_option("--k", action="store", type=int,
                  dest="k", default=5,
                  help='''Number of clusters to find. Default 5''')

    print(__doc__)
    (opts, args) = op.parse_args()
    if len(args) == 0:
        op.error("No input corpus provided!")
        sys.exit(1)

    if opts.algorithm not in supported_algorithms :
        op.error("Unsupported method " + opts.algorithm)
        op.print_help()
        sys.exit(1)

    if opts.linkage not in linkage_criteria :
        op.error("Unrecognized linkage criteria: " + opts.linkage)
        op.print_help()
        sys.exit(1)

    ## Convert the corpus into a matrix representation
    print "Loading corpus file: " + args[0]
    nTopFeatures = 10


    print 'Using the words from the article\'s text as features'
    documents, doc_term_matrix, vectorizer = tfidf_vectorize(WikiCorpus(open(args[0]), True))
    terms = vectorizer.get_feature_names()

    ## Perform the clustering
    if opts.algorithm == 'agglomerative' :
        clustering = agglomerative(doc_term_matrix, opts.k, opts.linkage)
    else :
        ## K-means or K-means++
        centroidsSelect = 'random' if opts.algorithm == 'k-means' else 'k-means++'
        clustering = kmeans(doc_term_matrix, k = opts.k, centroids = centroidsSelect)

    clusters2DocIds = clusterIdx2DocIdx(clustering)
    print("Summary of clusters: ")
    for clusterIdx in clusters2DocIds :
        print "Cluster %d" % clusterIdx

        docsInCluster = [documents[docIdx] for docIdx in clusters2DocIds[clusterIdx]]
        ## Sort the documents in the cluster by the number of words to get the most relevant documents
        docsInCluster.sort(key=lambda x : x.word_size, reverse=True)
        print("10 longest Wikipedia documents in the cluster: %s"  % (", ".join(doc.title for doc in docsInCluster[:nTopFeatures])))

        ## K-means allows us to show additional information
        if opts.algorithm != 'agglomerative' :
            order_centroids = clustering.cluster_centers_.argsort()[:, ::-1]
            ## Print the most representative features (words) of each cluster
            print("10 most frequent terms in the cluster: %s" % (", ".join(terms[x] for x in order_centroids[clusterIdx, :nTopFeatures])))


    if opts.algorithm != 'agglomerative' :
        # Calculate the sum of square error
        sseScore = sse(clustering, doc_term_matrix)
        print "Sum of square error: %.3f" %sseScore

    topics = get_categories(documents)

    if testPurity() :
        print "Your implementation of the purity metric seems correct."
        print "Calculating purity against categories " + str(topics)
        purity = purity(clustering, documents)
        print "Purity of the clustering: " + str(purity)
    else :
        print "The implementation of the purity metric seems to have a problem"

    if testEntropy() :
        print "Your implementation of the entropy metric seems correct."
        entropy = entropy(clustering, documents)
        print "Entropy of the clustering: " + str(entropy)
    else :
        print "The implementation of the entropy metric seems to have a problem"

    output(clustering, documents, opts.output)

    if 0:
        plotSSEvsK(doc_term_matrix)
