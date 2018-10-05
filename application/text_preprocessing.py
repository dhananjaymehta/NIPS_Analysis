"""

"""

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import re
from bs4 import BeautifulSoup
import unicodedata
import spacy
import pandas as pd

tokenizer = ToktokTokenizer()
nlp = spacy.load('en')
stopword_list = nltk.corpus.stopwords.words('english')


class TextCleaning:
    """
    This class has been created to pre_process data
    """
    #def __init__(self, dataframe):
    #    self.dataframe = dataframe

    def strip_html_tags(self, text):
        """

        :return:
        """
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text()
        return stripped_text

    def remove_accented_chars(self, text):
        """

        :return:
        """

        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def remove_special_characters(self, text, remove_digits=True):
        """

        :param remove_digits:
        :return:
        """
        pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
        text = re.sub(pattern, '', text)
        return text

    def simple_stemmer(self, text):
        """

        :param text:
        :return:
        """
        ps = nltk.porter.PorterStemmer()
        text = ' '.join([ps.stem(word) for word in text.split()])
        return text

    def lemmatize_text(self, text):
        """

        :param text:
        :return:
        """

        text = nlp(text)
        text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
        return text

    def remove_stopwords(self, text, is_lower_case=False):
        """

        :param text:
        :param is_lower_case:
        :return:
        """
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in stopword_list]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def normalize_corpus(self, doc, html_stripping=False, accented_char_removal=True, text_lower_case=True,
                         special_char_removal=True, stopword_removal=True, remove_digits=True,
                         text_lemmatization = False, text_stemming = False,):

        if html_stripping:
            doc = self.strip_html_tags(doc)

        # remove accented characters
        if accented_char_removal:
            doc = self.remove_accented_chars(doc)

        # lowercase the text
        if text_lower_case:
            doc = doc.lower()

        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)

        # remove special characters and\or digits
        if special_char_removal:
            # insert spaces between special characters to isolate them
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = self.remove_special_characters(doc, remove_digits=remove_digits)

        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)

        # remove stopwords:
        if stopword_removal:
            doc = self.remove_stopwords(doc, is_lower_case=text_lower_case)

        # lemmatize text:
        if text_lemmatization:
            doc = self.lemmatize_text(doc)

        # stemming text
        if text_stemming:
            doc = self.simple_stemmer(doc)
        return doc

    """
    def clean_data(self):        
        return (self.dataframe.Abstract.apply(self.normalize_corpus, args=(False, True, True, True, True, True, True)),  
                self.dataframe.PaperText.apply(self.normalize_corpus, args=(False, True, True, True, True, True, True)))

    """