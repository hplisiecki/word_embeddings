import pandas as pd
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
###############################################################################
corpus = pd.read_csv('C:/data/word2vec/term_9.csv')

texts = corpus.text.values

# preprocess the text
texts = [simple_preprocess(text) for text in texts]

model = Word2Vec(texts, min_count=1, vector_size=300, workers=4)

word = 'aborcja'
print(model.wv.most_similar(word))

word1 = 'aborcja'
word2 = 'mężczyzna'
print(model.wv.similarity(word1, word2))

# average vectors per document
doc_vecs = np.zeros((len(texts), 300))
for i, text in tqdm(enumerate(texts)):
    for word in text:
        doc_vecs[i] += model.wv.get_vector(word)
    doc_vecs[i] /= len(texts)

# average vectors per club
club_vecs = np.zeros((len(corpus.klub.unique()), 300))
for i, k in tqdm(enumerate(corpus.klub.unique())):
    print(k)
    club_vecs[i] = doc_vecs[corpus[corpus.klub == k].index].mean(axis=0)

club_dict = {k: v for k, v in zip(corpus.klub.unique(), club_vecs)}
del club_dict[np.nan]

# get vectors
X = np.array([club_dict[k] for k in club_dict.keys()])

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
# import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
plt.scatter(result[:, 0], result[:, 1])

words = list(club_dict.keys())
for i, word in enumerate(words):
	plt.annotate(word, xy=(result[i, 0], result[i, 1]))

