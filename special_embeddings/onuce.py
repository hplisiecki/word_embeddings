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
            date = row[1].date
            text = row[1].stemmed
            author = row[1].author
            id = row[1].id
            author_tag = 'PERSON_TAG' + '_' + author
            date_tag = 'DATE_TAG' + '_' + date
            doc_tag = 'DOC_TAG' + '_' + author + '_' + str(id) + '_' + date
            tokens = simple_preprocess(text)
            if self.bigram and self.trigram:
                self.words = self.trigram[self.bigram[tokens]]
            elif self.bigram and not self.trigram:
                self.words = self.bigram[tokens]
            else:
                self.words = tokens
            self.tags = [author_tag, doc_tag, date_tag]
            yield self.speeches(self.words, self.tags)

class phraseIterator(object):

    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        for row in tqdm(self.corpus.iterrows(), total=len(self.corpus)):
            # if party is not none
            text = row[1].stemmed
            yield simple_preprocess(text)


if __name__=='__main__':

    save_path = 'onuce/'

    corpus = pd.read_parquet(r'D:\PycharmProjects\Ukraina\data\disinformation_prepared.parquet')

    # # get the date of the earliest and latest tweet for each profile
    # earliest = []
    # latest = []
    # for name, group in corpus.groupby('username'):
    #     earliest.append(group.date.min())
    #     latest.append(group.date.max())
    #
    # # to csv
    # df = pd.DataFrame({'username': corpus.username.unique(), 'earliest': earliest, 'latest': latest})
    # usernames = df[['username']]
    # # save
    # usernames.to_csv('onuce_usernames.csv', index=False)




    # reset index
    corpus = corpus.reset_index(drop=True)
    corpus['date'] = [d[:7] for d in corpus.created]
    # dropna from stemmed_tweet
    # corpus = corpus.dropna(subset=['stemmed'])
    # corpus = corpus[~corpus.username.isin(['yoshiyamamo_to', 'entepe3', 'veritusasakano',
    #                                        'stranahan', 'kiranopal_', 'mandaryn62',
    #                                        'panasiukpiotr', 'lotemarco1', 'argan_beekan',
    #                                        'selianski'])]
    # corpus['length'] = corpus.stemmed.apply(lambda x: len(simple_preprocess(x)))
    # corpus[corpus.username == 'misiostary']

    # corpus = corpus[corpus.length > 5]
    # drop duplicates from text
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

