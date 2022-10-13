
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import pandas as pd
import pickle
# Official party colors.

current = 'onuce'

corpus = pd.read_parquet(r'D:\PycharmProjects\Ukraina\data\disinformation_prepared.parquet')
corpus = corpus.drop_duplicates(subset=['text'])

corpus['date'] = [str(d[:7]) for d in corpus.created]


person_labels = ['PERSON_TAG' + '_' + str(row[1]['author']) for row in corpus.iterrows()]
colors = ['red' if row[1]['onuce'] == 1 else 'blue' for row in corpus.iterrows()]
person_dict = {tag: tag.replace('PERSON_TAG_', '').split('_')[0] for tag in person_labels}
POL_COL_PERSON = {person_dict[person]: color for person, color in zip(person_labels, colors)}


document_labels = ['DOC_TAG_' + str(row[1]['author']) + '_' + str(row[1]['id']) + '_' + row[1]['date'] for row in corpus.iterrows()]
document_dict = {tag: tag.replace('DOC_TAG_', '').split('_')[0] for tag in document_labels}
POL_COL_DOC = {document_dict[doc]: color for doc, color in zip(document_labels, colors)}
# red if onuce and blue if not




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

