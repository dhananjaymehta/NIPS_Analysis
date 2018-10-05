"""
This module analyses the NIPS-2015 dataset. Analysis will include
1. creation of a Fasttext based word embedding to generate word representation
2. Identify major keywords for each document
3. Cluster the documents based on document similarity
"""

import pandas as pd
from pre_processing import TextCleaning
from task1 import WordEmbedding
from task2 import GenerateAspects
from task3 import DocumentClustering
import sys
import logging
import time
ts = time.strftime('%Y-%m-%d:%H:%M:%S')
logname = "./logs/" + str(ts) + ".log"
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filename=logname)


def main(file_path, is_html, is_stemming, ft_dim, n_cluster):
    """
    This method will implement the NIPS-2015 analysis. The analysis has following major steps -

    Step 1: This step involves reading the dataset provided by user from a given location

    Step 2: This will clean PaperText data and Abstract text data. New columns added to dataframe with cleaned data
    and a backup is created for cleaned data. User can enable/disable following steps - html_stripping, text_stemming

    Step 3: The next step is to generate the word-embedding using Fasttext. Once the data is cleaned it is used to
    generate a word-embedding model which can be used to get similar word-representations to a given word.
    User specify what dimension to be used in the fast-text model

    Step 4: Now identify keywords for each individual document

    Step 5: Using LDA cluster the documents into N-cluster. User can specify the number of clusters to be generated for documents

    :param file_path: file location
    :param is_html: boolean [T/F]
    :param is_stemming: boolean [T/F]
    :param ft_dim: range >100
    :param n_cluster:
    :return:
    """

    # Step 1: Read input dataset
    # ---------------------------------
    data1 = pd.read_csv(file_path)

    data=data1[:10]
    # taking subset of data with relevant columns - Id, Title, Abstract, PaperText
    data = data[['Id', 'Title', 'Abstract', 'PaperText']]

    # Step 2: Pre-processing the data
    # ----------------------------------
    pre_process_columns = TextCleaning()

    logging.info("--cleaning data--\n")
    data['AbstractClean'] = data['Abstract'].apply(pre_process_columns.normalize_corpus,
                                                       args=(is_html, True, True, True, True, True, True, is_stemming))
    data['PaperTextClean'] = data['PaperText'].apply(pre_process_columns.normalize_corpus,
                                                         args=(is_html, True, True, True, True, True, True, is_stemming))
    logging.info("--writing data for backup--\n")
    # Backup cleaned data
    data.to_csv("./input/cleaned_data.csv")

    # Step 3: Word embeddings - Fast-text
    # -------------------------------------
    logging.info("--generating word embeddings--\n")
    word_embedding = WordEmbedding(feature_size=int(ft_dim))
    word_embedding.build_model(data)

    # find word similarity for list of words
    words_similarity = ["classification","experiments"]
    logging.info(word_embedding.get_similar_words(words_similarity))

    # Step 4: Generate keywords for each doc
    # ---------------------------------------
    logging.info("--generating aspects--\n")
    get_keywords = GenerateAspects(data)
    get_keywords.gen_aspect()

    # Step 5: Generate document cluster
    # ---------------------------------------
    logging.info("--generating clusters--\n")
    gen_cluster = DocumentClustering(data)
    gen_cluster.get_document_cluster(topics_cnt=n_cluster)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
