#!/usr/bin/python3

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from sklearn.metrics.pairwise import euclidean_distances

def polarization_metric(model, country='USA'):

    M = model.vector_size
    dv = model.docvecs.index_to_key
    if country == 'Poland':
        parties = [d for d in dv if d.startswith('PARTY_TAG')]
        label_to_year = {1: '1991', 2: '1993', 3: '1997', 4: '2001', 5: '2005', 6: '2007', 7: '2011', 8: '2015', 9: '2019'}
        years = label_to_year.values()

    T = len(years)
    P = len(parties)
    z = np.zeros((P, M))
    for i in range(P):
        z[i,:] = model.docvecs[parties[i]]
    D = euclidean_distances(z)[0:T,T:P].diagonal().tolist()
    D = pd.DataFrame(D, columns=['euclidean_distance'])
    D['year'] = years
    return D[['year','euclidean_distance']]

