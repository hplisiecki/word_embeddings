#!/usr/bin/python3

import pkg_resources
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models.doc2vec import Doc2Vec
from special_embeddings.utils.labels import document_tags, person_tags, date_tags, party_tags
from special_embeddings.utils.guided import custom_projection_2D # for guided
from special_embeddings.utils.polarization import polarization_metric # done
from special_embeddings.utils.interpret import Interpret # done
from special_embeddings.utils.issues import issue_ownership
from sklearn.metrics.pairwise import cosine_similarity
from gensim import matutils

# MODEL_PATH = pkg_resources.resource_filename('partyembed', 'models/')

class Explore(object):

    def __init__(self, model, method='pca', dimensions=2, custom_lexicon=None, chamber=None, culprits = 'person', limit_party = None):
        self.model = model
        self.custom_lexicon = custom_lexicon
        self.M = self.model.vector_size
        self.reverse_dim1 = False; self.reverse_dim2 = False
        self.method = method
        self.culprits = culprits

        if self.culprits == 'document':
            self.fullnames, self.culprits, self.cols, self.mkers = document_tags(self.model)
            self.labels = [p.replace('DOC_TAG_', '') for p in self.culprits]

        elif self.culprits == 'person':
            self.fullnames, self.culprits, self.cols, self.mkers = person_tags(self.model, party = limit_party)
            self.labels = [p.replace('PERSON_TAG_', '') for p in self.culprits]

        elif self.culprits == 'date':
            self.fullnames, self.culprits, self.cols, self.mkers = date_tags(self.model)
            self.labels = [p.replace('DATE_TAG_', '') for p in self.culprits]

        elif self.culprits == 'party':
            self.fullnames, self.culprits, self.cols, self.mkers = party_tags(self.model)
            self.labels = [p.replace('PARTY_TAG_', '') for p in self.culprits]

        self.P = len(self.culprits)
        self.components = dimensions
        self.placement = self.dimension_reduction()

    def dimension_reduction(self):

        z=np.zeros(( self.P, self.M ))
        for i in range( self.P ):
            z[i,:] = self.model.dv[self.culprits[i]]
        if self.method=='pca':
            self.dr = PCA(n_components=self.components)
            self.Z = self.dr.fit_transform(z)
        elif self.method=='guided':
            self.Z = custom_projection_2D(z, self.model, custom_lexicon = self.custom_lexicon)
        else:
            raise ValueError("Model must be pca or guided.")
        Z = pd.DataFrame(self.Z)
        Z.columns = ['dim1', 'dim2']
        Z['party_label'] = self.labels

        return Z

    def create_dimension_vectors(self):
        component_vector_1 = np.zeros(self.M)
        component_vector_2 = np.zeros(self.M)
        for i in range(len(self.placement)):
            component_vector_1 += self.placement.dim1[i] * self.model.dv[self.culprits[i]]
            component_vector_2 += self.placement.dim2[i] * self.model.dv[self.culprits[i]]
        component_vector_1 = component_vector_1 / self.placement.dim1.sum()
        component_vector_2 = component_vector_2 / self.placement.dim2.sum()
        return component_vector_1, component_vector_2

    def compute_cosine_and_sort(self, idx, vector):
        sims = [np.dot(matutils.unitvec(a), matutils.unitvec(vector)) for a in self.model.dv[idx]]
        sims = np.array(sims)
        order = sims.argsort()[::-1]
        return sims, order

    def plot(self, axisnames=None, savepath=None, xlim=None, sample = None, labels = True):

        import matplotlib as mpl
        mpl.rcParams['axes.titlesize'] = 20
        mpl.rcParams['axes.labelsize'] = 20
        mpl.rcParams['font.size'] = 14
        if sample is not None:
            sampled_labels  = self.placement.party_label.sample(sample)
            sampled_dim1 = self.placement.dim1.sample(sample)
            sampled_dim2 = self.placement.dim2.sample(sample)
            sampled_cols = [self.cols[self.labels.index(l)] for l in sampled_labels]
        else:
            sampled_labels = self.placement.party_label
            sampled_dim1 = self.placement.dim1
            sampled_dim2 = self.placement.dim2
            sampled_cols = [self.cols[self.labels.index(l)] for l in sampled_labels]


        plt.figure(figsize=(10, 6))
        plt.scatter(sampled_dim1, sampled_dim2, color=sampled_cols)
        texts=[]
        if labels:
            for label, x, y, c in zip(sampled_labels, sampled_dim1, sampled_dim2, sampled_cols):
                plt.annotate(
                    label,
                    xy=(x, y), xytext=(-20, 20),
                    textcoords='offset points', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc=c, alpha=0.3),
                    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        if xlim:
            plt.xlim(xlim)
        if axisnames:
            plt.xlabel(axisnames[0])
            plt.ylabel(axisnames[1])
        else:
            if self.method=='guided':
                plt.xlabel("Economic Left-Right")
                plt.ylabel("Social Left-Right")
            else:
                plt.xlabel("Component 1")
                plt.ylabel("Component 2")
        if savepath:
            plt.savefig(savepath, dpi=600, bbox_inches='tight')
        plt.show()

    def plot_timeseries(self, dimension=1, axisnames=None, savepath=None, legend='upper left'):

        import matplotlib as mpl
        mpl.rcParams['axes.titlesize'] = 20
        mpl.rcParams['axes.labelsize'] = 20
        mpl.rcParams['font.size'] = 14

        reshaped = self.placement
        newvars = [label.split('_')[-1] for label in self.placement.party_label.values]
        reshaped['date'] = newvars
        reshaped['party'] = self.fullnames
        reshaped['color'] = self.cols
        fig, ax = plt.subplots(figsize=(22,15), sharex='all')

        if dimension==1:
            plt.scatter(reshaped.date, reshaped.dim1, color=reshaped.color)

        else:
            plt.scatter(reshaped.date, reshaped.dim2, color=reshaped.color)

        texts=[]
        for label, d1, d2, x,  c in zip(reshaped.party, reshaped.dim1, reshaped.dim2, reshaped.date, reshaped.color):
            if dimension == 1:
                y = d1
            else:
                y = d2
            plt.annotate(
                label,
                xy=(x, y), xytext=(-20, 20),
                textcoords='offset points', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc=c, alpha=0.3),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

        if axisnames:
            plt.xlabel(axisnames[0])
            plt.ylabel(axisnames[1])
        else:
            plt.xlabel("Date")
            if dimension==1:
                plt.ylabel("Ideological Placement (First Principal Component)")
            else:
                plt.ylabel("Second Principal Component")
        if savepath:
            plt.savefig(savepath, dpi=600, bbox_inches='tight')
        plt.show()

    def interpret(self, top_words=30, min_count=5, max_count = 1000000, max_features=1000000):
        Interpret(self.model, self.culprits, self.dr, self.placement, self.labels, \
                 min_count=min_count, max_count = max_count, rev1 = self.reverse_dim1, rev2 = self.reverse_dim2, \
                 max_features = max_features).top_words_list(top_words)

    def saturate(self, username = None, vector = None, returns = 'document', top = 10):

        if returns == 'words':
            if vector is not None:
                return self.model.wv.most_similar([vector], topn=top)
            elif username is not None:
                username = 'PERSON_TAG_' + username
                vector = self.model.dv[username]
                # most similar words
                return self.model.wv.most_similar(positive=[vector], topn=top)

        if returns == 'documents':
            documents = [d for d in self.model.dv.index_to_key if d.startswith('DOC_TAG')]
            if vector is not None:
                sims, order = self.compute_cosine_and_sort(documents, vector)
                return [(documents[s], sims[s]) for s in order[:top]]

            elif username is not None:
                vector = self.model.dv['PERSON_TAG_' + username]
                sims, order = self.compute_cosine_and_sort(documents, vector)
                return [(documents[s], sims[s]) for s in order[:top]]

        if returns == 'profiles':

            profiles = [d for d in self.model.dv.index_to_key if d.startswith('PERSON_TAG')]
            if vector is not None:
                sims, order = self.compute_cosine_and_sort(profiles, vector)
                return [(profiles[s], sims[s]) for s in order[:top]]

            elif username is not None:
                vector = self.model.dv['PERSON_TAG_' + username]
                sims, order = self.compute_cosine_and_sort(profiles, vector)
                return [(profiles[s], sims[s]) for s in order[:top]]

    def read_documents(self, docs):
        docs = [int(d.split('_')[-1]) for d in docs]
        corpus = pd.read_parquet(r'D:\PycharmProjects\Ukraina\data\onuce.parquet')
        corpus = corpus[corpus['id'].isin(docs)]
        return corpus

    def polarization(self):
        return polarization_metric(self.model)

    def issue(self, topic_word, lex_size=50):
        return issue_ownership(self.model, topic_word=topic_word, infer_vector=True, t_size=lex_size)

    def validate(self, custom_lexicon=None):
        if self.chamber:
            Validate(self.model, chamber=self.chamber, method=self.method, custom_lexicon=custom_lexicon).print_accuracy()
        else:
            Validate(self.model, method=self.method, custom_lexicon=custom_lexicon).print_accuracy()

    def benchmarks(self, test='analogies'):
        Validate(self.model, self.method).benchmarks(test=test)

save_path = 'special_embeddings/storage/'
model = Doc2Vec.load(save_path + 'doc2vec_0.model')

# compute cosine similarity
from sklearn.metrics.pairwise import euclidean_distances

# def cos_sim(a,b):
#     return cosine_similarity(a.reshape(1,-1), b.reshape(1,-1))[0][0]

# most similar words
# model.wv.most_similar


exploration = Explore(model, culprits='party')


exploration.interpret()

# exploration.plot(sample = 100, labels = False)

# exploration.plot(sample = 500, labels=False)


# v1, v2 = exploration.create_dimension_vectors()
# # v3 = v1 + v2
# # v4 = -v1 - v2
# #
# positive_profiles = exploration.saturate(vector = v1, returns='profiles', top=10)
# # profile =
# positive_profiles = [p[0].split('_')[-1] for p in positive_profiles]
# # _id = positive_profiles[0]
# # positive = f"https://twitter.com/intent/user?user_id={_id}&lang=en"
# #
# negative_profiles = exploration.saturate(vector = -v1, returns='profiles', top=10)
# # profile =
# negative_profiles = [p[0].split('_')[-1] for p in negative_profiles]
# _id = negative_profiles[0]
# negative = f"https://twitter.com/intent/user?user_id={_id}&lang=en"


# corpus = exploration.read_documents(document_names)

