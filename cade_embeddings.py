import pandas as pd
from cade.cade import CADE
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
from gensim.utils import simple_preprocess


aligner = CADE(size=300)

corpus = pd.read_csv('D:\\data\\workshops\\9_term.csv')

texts = corpus.text.values
texts = [simple_preprocess(text) for text in texts]

aligner.train_compass(texts, overwrite=False, save = True) # keep an eye on the overwrite behaviour
# load compass

for k in tqdm(set(corpus.klub.unique())):
   temp = corpus[corpus.klub == k]
   texts = temp.text.values
   if len(texts) > 1:
      texts = [simple_preprocess(text) for text in texts]
      slice = aligner.train_slice(texts, f'club_{k}', train_vocab=False, save=True)

pis = Word2Vec.load("model/club_PiS.model")
ko =  Word2Vec.load("model/club_KO.model")
lewica = Word2Vec.load("model/club_Lewica.model")
word = 'aborcja'
print(pis.wv.most_similar(word))
print(ko.wv.most_similar(word))
print(lewica.wv.most_similar(word))

vec_pis = pis.wv[word]
vec_ko = ko.wv[word]
vec_lewica = lewica.wv[word]
from scipy.spatial.distance import cosine
print(1 - cosine(vec_ko, vec_lewica))