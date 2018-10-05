"""
This module generates word-embedding using Fst-Text.
This module will handle the case of word-representation
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
    """
    implement the word embedding
    """
    def __init__(self, feature_size=100, window_context=50, min_word_count=5, sample=1e-3,mode=1):
        """
        i
        """
        self.model = None                       # Model to be generated
        self.feature_size = feature_size        # Word vector dimensionality
        self.window_context = window_context    # Context window size
        self.min_word_count = min_word_count    # Minimum word count
        self.sample = sample                    # Downsample setting for frequent words
        self.mode = mode                        # sg decides whether to use the skip-gram model (1) or CBOW (0)

    def build_model(self, data):
        """
        This will build a model using Fast-text
        :param data: dataframe as an input
        :return:
        """
        # tokenize the text
        tokenized_corpus = [wpt.tokenize(document) for document in list(data.PaperTextClean)]

        # Taking a backup store to text file
        tokenized_corpus_backup = [[str(id), tokenized_doc] for id, tokenized_doc in zip(list(data.Id), tokenized_corpus)]

        with open('./output/tokenized_text.txt', 'w') as f:
            for item in tokenized_corpus_backup:
                rec = '##'.join(str(_item) for _item in item)
                f.write("%s\n" % rec)

        # building a fasttext word-embedding model
        self.model = FastText(tokenized_corpus, size=self.feature_size, window=self.window_context,
                            min_count=self.min_word_count,sample=self.sample, sg=self.mode, iter=50)

        # save model after its been created
        self.model.save('./model/task1/nips_fasttext')

        # Load model if needed for retraining
        # model = FastText.load(name)

    def get_similar_words(self,search_words):
        """
        This function will look for similar words for input words from users
        :param search_words: (list), list of words that user want to search
        :return: list of similar words
        """
        similar_words = {}
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
        This function will generate the visualization for similarity of words from previous function
        :param similar_words: (dict), map of related words generated from get_similar_words
        :return: plot of words similarity
        """
        if not similar_words:
            return

        words = sum([[k] + v for k, v in similar_words.items()], [])
        wvector = self.model.wv[words]

        pca = PCA(n_components=2)
        np.set_printoptions(suppress=True)
        plot = pca.fit_transform(wvector)
        labels = words

        plt.figure(figsize=(18,12))
        plt.scatter(plot[:, 0], plot[:, 1], c='lightgreen', edgecolors='g')
        for label, x, y in zip(labels, plot[:, 0], plot[:, 1]):
            plt.annotate(label, xy=(x+0.01, y+0.01), xytext=(0, 0), textcoords='offset points')

        name="./visualizations/FastText/viz_fasttext"
        plt.savefig( name+".png")