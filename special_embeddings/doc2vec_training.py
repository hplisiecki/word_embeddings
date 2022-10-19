import pandas as pd
from gensim.utils import simple_preprocess
import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.models.phrases import Phrases, Phraser
from gensim import corpora
from collections import namedtuple
import logging
from tqdm import tqdm
import numpy as np
# Create document embeddings

# word2vec = KeyedVectors.load('our_vectors.kv')



class corpusIterator(object):

    def __init__(self, corpus, bigram=None, trigram=None):
        if bigram:
            self.bigram = bigram
        else:
            self.bigram = None
        if trigram:
            self.trigram = trigram
        else:
            self.trigram = None
        self.corpus = corpus

    def __iter__(self):
        self.speeches = namedtuple('speeches', 'words tags')
        for row in self.corpus.iterrows():
            # if party is not none
            month = row[1].month
            text = row[1].text
            author = row[1].fullname
            party = row[1].klub
            author_tag = 'PERSON_TAG' + '_' + author.replace(' ', '_')
            date_tag = 'MONTH_TAG' + '_' + month
            party_tag = 'PARTY_TAG' + '_' + party

            tokens = simple_preprocess(text)
            if self.bigram and self.trigram:
                self.words = self.trigram[self.bigram[tokens]]
            elif self.bigram and not self.trigram:
                self.words = self.bigram[tokens]
            else:
                self.words = tokens
            self.tags = [author_tag, date_tag, party_tag]
            yield self.speeches(self.words, self.tags)

class phraseIterator(object):

    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for row in tqdm(self.corpus.iterrows(), total=len(self.corpus)):
            # if party is not none
            text = row[1].text
            yield simple_preprocess(text)


if __name__=='__main__':

    save_path = 'storage/'

    corpus = pd.read_csv('D:\\data\\workshops\\9_term.csv')



    # dropna from klub
    corpus = corpus.dropna(subset=['klub'])
    # reset index
    corpus = corpus.reset_index(drop=True)
    corpus['month'] = [d[:7] for d in corpus.data]
    corpus = corpus.drop_duplicates(subset=['text'])


    print('corpus loaded')

    phrases = Phrases(phraseIterator(corpus))
    bigram = Phraser(phrases)
    print('bigram done')

    tphrases = Phrases(bigram[phraseIterator(corpus)])
    trigram = Phraser(tphrases)
    print('trigram done')

    bigram.save(save_path + 'phraser_bigrams')
    trigram.save(save_path + 'phraser_trigrams')
    #
    # load
    bigram = Phraser.load(save_path + 'phraser_bigrams')
    trigram = Phraser.load(save_path + 'phraser_trigrams')

    print('phraser loaded')

    model0 = Doc2Vec(vector_size=300, window=5, min_count=50, workers=8, epochs=5)

    # , bigram=bigram, trigram=trigram
    model0.build_vocab(corpusIterator(corpus, bigram=bigram, trigram=trigram))
    print('vocab done')

    model0.train(corpusIterator(corpus, bigram=bigram, trigram=trigram), total_examples=model0.corpus_count, epochs=model0.epochs)
    print('training done')

    model0.save(save_path + 'doc2vec_0.model')
    print('model saved')

    # load

    # load

