"""

"""
from __future__ import division
from collections import defaultdict
import spacy
import nltk
import json

# nlp = spacy.load('en')
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(nlp.create_pipe('sentencizer'))

# list of POS tag belonging to nouns
noun_tag = set(['NN', 'NNP', 'NNS'])

# keywords to avoid
nonaspects = set()


class Ngrams:
    """
    Class of functions for extracting Unigrams and Bigrams
    """

    def __init__(self, document):
        self.text = document
        self.unigrams = defaultdict()
        self.bigrams = defaultdict()

    def unigram(self):
        """
        Iterates through each token of spacy sentence and collects lemmas of all nouns into a set.
        :param sent:
        :return: set
        """
        # convert self.text to 'spacy.tokens.doc.Doc'format
        for token in nlp(self.text):
            # filter to only consider nouns, valid aspects, and uncommon words
            if token.tag_ in noun_tag and token.lemma_ not in nonaspects:
                if token.lemma_ in self.unigrams:
                    self.unigrams[token.lemma_] += 1
                else:
                    self.unigrams[token.lemma_] = 1

        # Note: We need to normalize the count of nouns with respect to the total noun counts
        # as in certain scenarios there will be few nouns and we can't pick nouns based on count
        nouns_count = sum([val for val in self.unigrams.values()])

        wordset_norm = {key: val / nouns_count for (key, val) in self.unigrams.items()}
        unigram_keywords = [u_keyword for u_keyword in wordset_norm.keys() if wordset_norm[u_keyword] > 0.01]

        return unigram_keywords

    def bigram(self):
        """
        :return:
        """
        pos_tags = nltk.pos_tag(nltk.word_tokenize(self.text))

        def ngramise(sequence):
            """
            generate bigrams phrases
            """
            for bgram in nltk.ngrams(sequence, 2):
                yield bgram

        for ngram in ngramise(pos_tags):
            tokens, tags = zip(*ngram)
            # I am looking for bigram pairs of Adjective + Noun, this relationship is extensible to Noun + Adjective or
            # pronoun + noun etc.
            if tags == ('JJ', 'NN'):
                b_key = tokens[0]+" "+tokens[1]
                if tokens in self.bigrams:
                    self.bigrams[b_key] += 1
                else:
                    self.bigrams[b_key] = 1

        # I am not returning the bigrams after normalization as the bigram pairs are rare
        #bigram_keywords = [b_keyword for b_keyword in self.bigrams.keys() if self.bigrams[b_keyword] > 1]
        #return bigram_keywords
        return list(self.bigrams.keys())


class GenerateAspects:
    """
    Uses spacy to parse and split the sentences
        Return number of reviews, sentences, and list of spacy objects
    """

    def __init__(self, data):
        self.word = 0
        self.output = list()
        self.data = data
        self.keywords = dict()

    def gen_aspect(self):
        # get list of all documents id
        paper_id = list(self.data['Id'])

        # generate unigrams and bigrams for each p_id: paper_id
        for p_id in paper_id:
            # p_text: text of paper,
            p_text = str(self.data[self.data.Id == p_id]['PaperTextClean'].iloc[0])
            self.keywords[p_id] = {'unigrams':Ngrams(p_text).unigram(), 'bigrams':Ngrams(p_text).bigram()}

        # output results to json
        self.write_json_result()


    def write_json_result(self):
        """
        This function will write the keywords to output json file
        :return:
        """
        with open('./output/keywords.json', 'w') as fp:
            json.dump(self.keywords, fp)

