"""
This module implements Latent Dirichlet allocation for clustering the documents.
Number of topics generated are based on user inputs
"""
from gensim import corpora, models, similarities
from itertools import chain
import json


class DocumentClustering:
    """
    Implement LDA for clustering the documents into N-topics.
    This will generate -
    documents_cluster: map of cluster_group_id to document_id
    cluster_word_map: map of cluster_group_id to top K significant words in cluster
    """
    def __init__(self, data):
        """
        Initialize the model, cluster_word_map, document_cluster
        :param data:
        """
        self.data = data
        self.lda_model = None
        self.documents_cluster = dict()
        self.cluster_word_map = dict()

    def _gen_document_cluster(self, topic_count=5):
        """
        Generate text corpus for all document_id and paper_text, keep words that will have length >3.
        Next, generate word tokens. Create dictionary of word tokens and generate LDA model.
        Finally, cluster the documents into topic based on threshold

        :param topic_count: Count of cluster that will be generated
        :return:
        """
        document_corpus = [(_id, _text) for _id, _text in zip(self.data.Id, self.data.PaperTextClean)]

        # Including words with minimum length of 3
        # This will avoid cases where pre-processing removed special characters and numbers
        # Ex: pg13 -> pg, a123b-> ab, Q1_2018->Q
        document_updated = [[word for word in document[1].lower().split() if len(word) > 3]
                            for document in document_corpus]

        all_tokens = sum(document_updated, [])

        # Removing words that appear only once
        tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
        texts = [[word for word in text if word not in tokens_once] for text in document_updated]

        id2word = corpora.Dictionary(texts)
        bag_of_words = [id2word.doc2bow(text) for text in texts]

        self.lda_model = models.ldamodel.LdaModel(corpus=bag_of_words, id2word=id2word,
                                                  num_topics=topic_count, update_every=1, chunksize=10000, passes=1)

        lda_corpus = self.lda_model[bag_of_words]

        # Find threshold for document to be part of cluster, threshold to be 1/#clusters,
        # Average the sum of all probabilities:
        scores = list(chain(*[[score for topic_id, score in topic]
                              for topic in [doc for doc in lda_corpus]]))

        threshold = sum(scores) / len(scores)

        #self.lda_model.save('lda_model')

        # Generate document cluster: {cluster_id: [list of documents]}
        for _doc in range(len(document_corpus)):
            for _lda in lda_corpus[_doc]:
                if _lda[1] > threshold:
                    key = _lda[0]
                    if key in self.documents_cluster:
                        self.documents_cluster[_lda[0]].append(document_corpus[_doc][0])
                    else:
                        self.documents_cluster[_lda[0]] = [document_corpus[_doc][0]]

        # cluster_word_map: cluster_id: [(word, significance of word)]}
        _cluster_word_map = {key: self.lda_model.show_topic(key, topn=10)
                             for key in range(self.lda_model.num_topics)}

        self.cluster_word_map = {cluster_id: [word[0] for word in words] for cluster_id, words in _cluster_word_map.items()}

    def get_document_cluster(self, topics_cnt):
        """
        Cluster documents, write resulting documents_cluster and cluster_map to
        :param topics_cnt: number of topics that are needed
        :return:
        """
        self._gen_document_cluster(topics_cnt)

        with open('./output/task3/document_cluster.json', 'w') as f:
            json.dump(self.documents_cluster, f)

        with open('./output/task3/cluster_map.json', 'w') as f:
            json.dump(self.cluster_word_map, f)