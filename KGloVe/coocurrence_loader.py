import pathlib
import pickle
from struct import unpack
from typing import BinaryIO, List, Optional, Tuple, cast
from sklearn import cluster
import pandas as pd
import sys
import numpy as np
import scipy.sparse


def _read_little_endian_crec(file: BinaryIO
                             ) -> Optional[Tuple[int, int, float]]:
    le_int = file.read(16)
    # https://docs.python.org/3/library/struct.html#format-strings
    if len(le_int) == 0:
        return None
    crec = cast(Tuple[int, int, float], unpack('<iid', le_int))
    return crec


def load(cooccurrence_file_content: BinaryIO) -> scipy.sparse.coo_matrix:

    row: List[int] = []
    column: List[int] = []
    data: List[float] = []
    while (cooccurrence_file_content.readable()):
        crec = _read_little_endian_crec(cooccurrence_file_content)
        if crec is None:
            break
        row.append(crec[0])
        column.append(crec[1])
        data.append(crec[2])
    result = scipy.sparse.coo_matrix((data, (row, column)), dtype=np.float64)
    return result

def save_top_nodes(nodelist, filename):
    top_nodes = []
    for j in range(10):  # Pick ten nodes from each cluster, only one should be used. The others are back-up in case the selected node is filtered out (in case of multiple selection strategies)
        for i in range(n_clusters):
            if j < len(np.where(nodelist == i)[0]):
                index = np.where(nodelist == i)[0][j]
                # now match this index to node value:
                raw_value = dict.get(index)
                value = int(raw_value.strip('<').strip('>'))
                top_nodes.append((value,j))  # top_nodes should be of the form [(9677, 1), (2523, 1), ...]
    sys.stdout = original_stdout  
    with open(filename, "wb") as fp:  # Pickling
        sys.stdout = fp
        pickle.dump(top_nodes, fp)
        sys.stdout = original_stdout # Reset the standard output to its original value  
    return top_nodes
    
def save_diff_nodes(nodelist, filename):
    diff_nodes = []
    for k in range(10): 
        # make a dictionary that contains the cluster label and the frequency of that label
        dict_freq = {}  # key is element, value is frequency
        for i in range(len(nodelist)):
            if nodelist[i] in dict_freq:
                freq = dict_freq[nodelist[i]]
                freq += 1
                dict_freq[nodelist[i]] = freq
            else:
                dict_freq.update({nodelist[i]: 1})
        counter = 0
        while counter < n_clusters:
            highest_value = max(dict_freq, key=dict_freq.get) # pick cluster with highest frequence
            indexes = np.where(nodelist == highest_value)[0]  # use all indexes in that cluster
            if len(indexes) > n_clusters: 
                for j in range(n_clusters):  # only select as many anchors as needed
                    raw_value = dict.get(indexes[j])
                    value = int(raw_value.strip('<').strip('>')) # convert indexes to nodes
                    diff_nodes.append((value, k))      
            else:
                for index in indexes:
                    raw_value = dict.get(index)
                    value = int(raw_value.strip('<').strip('>'))
                    diff_nodes.append((value, k))  # top_nodes should be of the form [(9677, 1), (2523, 1), ...]
            del dict_freq[highest_value]
            counter += len(indexes)

    with open(filename, "wb") as fp:  # Pickling
        sys.stdout = fp
        pickle.dump(diff_nodes, fp)
        sys.stdout = original_stdout # Reset the standard output to its original value  
    return diff_nodes
    
    

def create_index_dict():
    dict = {}   #key is index, value is node name
    index = 0
    lines = open("/home/diependaal/KGlove/testInput/wn18rr.nt").readlines()   # Input file containing the knowledge graph triples
    for line in lines:
        line = line.split(' ')
        if line[0] not in dict.values():
            dict.update({index: line[0]})
            index += 1
        if line[2] not in dict.values():
            dict.update({index: line[2]})
            index += 1
    #print(dict)
    return dict

if __name__ == "__main__":
    original_stdout = sys.stdout # Save a reference to the original standard output
    p = pathlib.Path("/home/diependaal/KGlove/output/glove_input_file-testInput_wn18rr_nt-no_literals-forwardWeigher_UniformWeigher-alpha_0.69999999999999996-eps_1.0000000000000001e-05-onlyEntities_no-edges_yes.bin")
    with open(p, 'rb') as file:
        m = load(file)
    
    array = m.tocsc()
    n_clusters = 500
        # wn18rr has 500 anchor nodes
        # fbb15k237 has 1000 anchor nodes
    dict = create_index_dict()
    
    # K-means clustering
    k_means = cluster.KMeans(n_clusters=n_clusters)
    clusterlabelslist = k_means.fit_predict(array)    # use fit_predict
    save_top_nodes(clusterlabelslist, "/home/diependaal/KGlove/wn-k-means-500")
    save_diff_nodes(clusterlabelslist, "/home/diependaal/KGlove/wn-diff-k-means-500")
    

    #Agglomerative clustering, does not take sparse graph. 
    array = array.toarray()
    agglomerative = cluster.AgglomerativeClustering(n_clusters=n_clusters, linkage="single") 
    clusterlabelslist = agglomerative.fit_predict(array)    
    save_top_nodes(clusterlabelslist, "/home/diependaal/KGlove/wn-agglo-500")
    Ssave_diff_nodes(clusterlabelslist, "/home/diependaal/KGlove/wn-diff-agglo-500")

    
   
