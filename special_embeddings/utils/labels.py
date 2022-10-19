
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import pandas as pd
import pickle
# Official party colors.

current = 'onuce'

corpus = pd.read_csv('D:\\data\\workshops\\9_term.csv')
corpus = corpus.dropna(subset=['klub'])
# reset index
corpus = corpus.reset_index(drop=True)
corpus['month'] = [d[:7] for d in corpus.data]
corpus = corpus.drop_duplicates(subset=['text'])


person_labels = ['PERSON_TAG' + '_' + str(row[1]['fullname'].replace(' ', '_')) for row in corpus.iterrows()]
person_dict = {tag: fn for tag, fn in zip(person_labels, corpus.fullname)}

party_labels = ['PARTY_TAG' + '_' + str(row[1]['klub']) for row in corpus.iterrows()]
party_dict = {tag: fn for tag, fn in zip(party_labels, corpus.klub)}

# document_labels = ['DOC_TAG_' + str(row[1]['username']) + '_' + str(row[1]['id']) + '_' + row[1]['date'] for row in corpus.iterrows()]
# document_dict = {tag: tag.replace('DOC_TAG', '').split('_')[0] for tag in document_labels}


import seaborn as sns

palette = sns.color_palette(None, len(set(person_dict.values())))
POL_COL_PERSON = {person: palette[i] for i, person in enumerate(set(person_dict.values()))}

palette = sns.color_palette(None, len(set(party_dict.values())))
POL_COL_PARTY = {party: palette[i] for i, party in enumerate(set(party_dict.values()))}


# palette = sns.color_palette(None, len(set(document_dict.values())))
# POL_COL_DOC = {doc: palette[i] for i, doc in enumerate(set(document_dict.values()))}


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

def party_tags(model, grayscale=False):

    parties = [d for d in model.docvecs.index_to_key if d.startswith('PARTY_TAG')]
    fullnames = [party_dict[d] for d in parties]
    cols = [POL_COL_PARTY[f] for f in fullnames]
    mkers = ['o' for i in range(len(parties))]

    return (fullnames, parties, cols, mkers)

def document_tags(model, grayscale=False):

    documents = [d for d in model.docvecs.index_to_key if d.startswith('DOC_TAG_')]
    fullnames = [document_dict[d] for d in documents]
    cols = [POL_COL_DOC[f] for f in fullnames]
    mkers = ['o' for i in range(len(documents))]

    return (fullnames, documents, cols, mkers)

