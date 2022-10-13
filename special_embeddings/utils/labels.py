
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import pandas as pd
import pickle
# Official party colors.

current = 'onuce'

corpus = pd.read_parquet(r'D:\PycharmProjects\Ukraina\data\onuce.parquet')
corpus = corpus.dropna(subset=['stemmed_tweet'])

corpus.date = [str(d[:7]) for d in corpus.date]

person_labels = ['PERSON_TAG' + '_' + str(row[1]['username']) for row in corpus.iterrows()]
person_dict = {tag: tag.replace('PERSON_TAG', '').split('_')[0] for tag in person_labels}

document_labels = ['DOC_TAG_' + str(row[1]['username']) + '_' + str(row[1]['id']) + '_' + row[1]['date'] for row in corpus.iterrows()]
document_dict = {tag: tag.replace('DOC_TAG', '').split('_')[0] for tag in document_labels}


if current == 'war':
    # corpus = pd.read_csv(r'D:\PycharmProjects\Ukraina\data\war_text_and_author.csv')

    # person_labels = ['PERSON_TAG' + '_' + str(row[1]['author']) for row in corpus.iterrows()]
    # person_dict = {tag: tag.replace('PERSON_TAG', '').split('_')[0] for tag in person_labels}
    # import pickle
    # # save
    # with open(r'D:\PycharmProjects\Ukraina\special_embeddings\war\person_dict.pickle', 'wb') as handle:
    #     pickle.dump(person_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # load
    with open(r'D:\PycharmProjects\Ukraina\special_embeddings\war\person_dict.pickle', 'rb') as handle:
        person_dict = pickle.load(handle)

    # document_labels = ['DOC_TAG_' + str(row[1]['author']) + '_' + str(row[1]['id']) + '_' + str(row[1]['date'][:7]) for row in corpus.iterrows()]
    # document_dict = {tag: tag.replace('DOC_TAG', '').split('_')[0] for tag in document_labels}
    #
    # # save
    # with open(r'D:\PycharmProjects\Ukraina\special_embeddings\war\document_dict.pickle', 'wb') as handle:
    #     pickle.dump(document_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # load
    with open(r'D:\PycharmProjects\Ukraina\special_embeddings\war\document_dict.pickle', 'rb') as handle:
        document_dict = pickle.load(handle)

import seaborn as sns

palette = sns.color_palette(None, len(set(person_dict.values())))
POL_COL_PERSON = {person: palette[i] for i, person in enumerate(set(person_dict.values()))}

palette = sns.color_palette(None, len(set(document_dict.values())))
POL_COL_DOC = {doc: palette[i] for i, doc in enumerate(set(document_dict.values()))}


def party_labels():
    pass

    # if country == 'Poland':
        # modify to set custom labels (needs to be a dictionary that takes party tags and returns a labels)

def person_tags(model, grayscale=False, party=False):

    if party:
        collate = []
        for p in party:
            mps = [d for d in model.docvecs.index_to_key if d.startswith('PERSON_TAG')]
            mps = [m for m in mps if '_' + p + '_' in m]
            collate.extend(mps)
        mps = collate
    else:
        mps = [d for d in model.docvecs.index_to_key if d.startswith('PERSON_TAG')]
    fullnames = [person_dict[d] for d in mps]
    cols = [POL_COL_PERSON[f] for f in fullnames]
    mkers = ['o' for i in range(len(mps))]

    return (fullnames, mps, cols, mkers)

def date_tags(model, grayscale=False):

    dates = [d for d in model.docvecs.index_to_key if d.startswith('DATE_TAG')]
    cols = ['black' for i in range(len(dates))]
    mkers = ['o' for i in range(len(dates))]

    return (dates, dates, cols, mkers)



def document_tags(model, grayscale=False):

    documents = [d for d in model.docvecs.index_to_key if d.startswith('DOC_TAG_')]
    fullnames = [document_dict[d] for d in documents]
    cols = [POL_COL_DOC[f] for f in fullnames]
    mkers = ['o' for i in range(len(documents))]

    return (fullnames, documents, cols, mkers)

