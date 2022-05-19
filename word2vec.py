from gensim.models import Word2Vec

data_directory = 'C:/data/word2vec'
# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]
#####################
# train model
model = Word2Vec(sentences, min_count=1)
# summarize the loaded model
print(model)
# summarize vocabulary
words = list(model.wv.key_to_index.keys())
print(words)
# access vector for one word
word = 'sentence'
print(model.wv.get_vector(word))
# save model
model.save(data_directory + '/model.bin')
# load model
new_model = Word2Vec.load(data_directory + '/model.bin')
print(new_model)







# plotting embeddings
# get all embeddings
X = model.wv.vectors
# import PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
# import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
plt.scatter(result[:, 0], result[:, 1])

words = list(model.wv.key_to_index.keys())
for i, word in enumerate(words):
	plt.annotate(word, xy=(result[i, 0], result[i, 1]))

