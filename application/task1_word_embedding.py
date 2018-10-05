"""
https://radimrehurek.com/gensim/models/fasttext.html
"""
from gensim.test.utils import get_tmpfile
from sklearn.decomposition import PCA
import numpy as np
import nltk
from gensim.models.fasttext import FastText
import matplotlib.pyplot as plt
import spacy

nlp = spacy.load('en')
wpt = nltk.WordPunctTokenizer()


class WordEmbedding:
    def __init__(self,feature_size=100, window_context=50, min_word_count=5, sample=1e-3,mode=1):
        """

        :param feature_size:
        :param window_context:
        :param min_word_count:
        :param sample:
        :param mode:
        """
        self.model = None                       # Model to be generated
        self.feature_size = feature_size        # Word vector dimensionality
        self.window_context = window_context    # Context window size
        self.min_word_count = min_word_count    # Minimum word count
        self.sample = sample                    # Downsample setting for frequent words
        self.mode = mode                        # sg decides whether to use the skip-gram model (1) or CBOW (0)

    def build_model(self,data):
        """
        :param data:
        :return:
        """
        # tokenize the text - abstract
        tokenized_corpus = [wpt.tokenize(document) for document in list(data.AbstractClean)]

        # Taking a backup store to text file
        tokenized_corpus_backup = [[str(id), tokenized_doc] for id, tokenized_doc in zip(list(data.Id), tokenized_corpus)]

        # take a backup of tokeninzed_text -
        with open('./output/tokenized_text.txt', 'w') as f:
            for item in tokenized_corpus_backup:
                rec = '##'.join(str(_item) for _item in item)
                f.write("%s\n" %rec)

        # building a fasttext vectorizer model
        self.model = FastText(tokenized_corpus, size=self.feature_size, window=self.window_context,
                            min_count=self.min_word_count,sample=self.sample, sg=self.mode, iter=50)

        # save model after its been created
        self.model.save('./model/nips_fasttext')

        # this model can be recalled in later stages and we can continue to train the model
        # model = FastText.load(fname)

    def get_similar_words(self,search_words):
        """
        This function will look for similar words for input words from users
        :param search_words: (list), list of words that user want to search
        :return: list of similar words
        """
        #search_words=["classification"]
        similar_words={}
        if not search_words:
            return "Please enter words to be searched"
        else:
            for search_term in search_words:
                # view similar words based on gensim's FastText model
                similar_words = {search_term: [item[0] for item in self.model.wv.most_similar([search_term], topn=10)]}

        self.get_word_viz(similar_words)
        return similar_words

    def get_word_viz(self,similar_words):
        """
        This function will generate the visualization for similarity of words
        :param similar_words: (dict), map of related words generated from get_similar_words
        :return: plot of words similarity
        """
        if not similar_words:
            return

        words = sum([[k] + v for k, v in similar_words.items()], [])
        wvs = self.model.wv[words]

        pca = PCA(n_components=2)
        np.set_printoptions(suppress=True)
        P = pca.fit_transform(wvs)
        labels = words

        plt.figure(figsize=(18,12))
        plt.scatter(P[:, 0], P[:, 1], c='lightgreen', edgecolors='g')
        for label, x, y in zip(labels, P[:, 0], P[:, 1]):
            plt.annotate(label, xy=(x+0.01, y+0.01), xytext=(0, 0), textcoords='offset points')

        #plt.show()
        name="./visualizations/viz_fasttext"
        plt.savefig( name+".png")