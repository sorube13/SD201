puntos = ['p1', 'p2', 'p3', 'p1']
labels2docs = {}
n_documents = len(puntos)
for i in range(n_documents):
    labeli = puntos[i]
    if labeli not in labels2docs:
        labels2docs[labeli] = []
    labels2docs[labeli].append(i)

histogram = {}
totalSum = 0.0
for cluster_index in labels2docs:
    if cluster_index not in histogram :
        histogram[cluster_index] = [] ##Initialize the list
    histogram[cluster_index] = (len(labels2docs[cluster_index]))
    totalSum += len(labels2docs[cluster_index])
max_cluster = max(histogram, key=histogram.get)

purity = histogram[max_cluster]/totalSum
import pdb; pdb.set_trace()
