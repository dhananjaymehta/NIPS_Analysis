"""
This module identify keywords for a document. This module identify both unigrams and bigrams.
A Noun qualifies for being a keyword as a sentence revolves around the noun.
Unigrams are identified with high success rate but bigrams are not very accurate.
Lot of Bigrams are generated but it is difficult to identify key bigrams as bigram pairs are unique
"""
from __future__ import division
from collections import defaultdict
import spacy
import nltk
import json

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(nlp.create_pipe('sentencizer'))

# list of POS tag belonging to nouns
noun_tag = {'NN', 'NNP', 'NNS'}

# keywords to avoid
nonaspects = set()


class Ngrams:
    """
    This class identify unigrams and bigrams in the document
    """

    def __init__(self, document):
        self.text = document
        self.unigrams = defaultdict()
        self.bigrams = defaultdict()

    def unigram(self):
        """
        To identify unigrams, iterate over all the tokens. Identify the nouns as they are candidate keywords.
        Check if the keyword was in the exclusion list. Take a count of individual keywords and
        return the keywords above threshold

        :return: list of important unigram keywords
        """
        for token in nlp(self.text):
            # only consider nouns, valid aspects, and uncommon words
            if token.tag_ in noun_tag and token.lemma_ not in nonaspects:
                if token.lemma_ in self.unigrams:
                    self.unigrams[token.lemma_] += 1
                else:
                    self.unigrams[token.lemma_] = 1

        # Note: We need to normalize the count of nouns with respect to the total noun counts
        all_nouns_count = sum([val for val in self.unigrams.values()])

        threshold = 0.01

        wordset_normalized = {key: val / all_nouns_count for (key, val) in self.unigrams.items()}
        unigram_keywords = [u_keyword for u_keyword in wordset_normalized.keys()
                            if wordset_normalized[u_keyword] > threshold]

        return unigram_keywords

    def bigram(self):
        """
        Currently bigram pairs of Adjective + Noun (JJ+NN) are being identified. This relationship can be extended to
        Noun + Adjective or pronoun + noun etc.  Method returns the bigrams without any normalization as
        bigram pairs are rare

        :return: list of bigrams
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
            if tags == ('JJ', 'NN'):
                b_key = tokens[0]+" "+tokens[1]
                if tokens in self.bigrams:
                    self.bigrams[b_key] += 1
                else:
                    self.bigrams[b_key] = 1

        return list(self.bigrams.keys())


class GenerateAspects:
    """
    Generate bigram and unigram keywords for each document. Store the result in a json based file -
    {paper_id:
        unigrams : [list]
        bigrams : [list]
    }
    """
    def __init__(self, data):
        self.word = 0
        self.output = list()
        self.data = data
        self.keywords = dict()

    def gen_aspect(self):
        """

        :return:
        """
        paper_id = list(self.data['Id'])

        for p_id in paper_id:
            # p_text: cleaned data for paper text, the text is then converted to Spacy NLP format
            p_text = str(self.data[self.data.Id == p_id]['PaperTextClean'].iloc[0])
            self.keywords[p_id] = {'unigrams':Ngrams(p_text).unigram(), 'bigrams':Ngrams(p_text).bigram()}

        self.write_json_result()

    def write_json_result(self):
        """
        This function will write the keywords to output json file
        :return:
        """
        with open('./output/task2/keywords.json', 'w') as fp:
            json.dump(self.keywords, fp)

