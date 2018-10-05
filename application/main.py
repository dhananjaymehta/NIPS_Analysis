"""
This module analyses the NIPS-2015 dataset. Analysis will include
1. creation of a Fasttext based word embedding to generate word representation
2. Identify major keywords for each document
3. Cluster the documents based on document similarity
"""

import pandas as pd
import sys
from pre_processing import TextCleaning
from task1 import WordEmbedding
from task2 import GenerateAspects
from task3 import DocumentClustering
import sys


def main(argv):
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

    :param argv: user_defined parameters - 0: file location, 1: is_html, 2:text_stemming, 3:ft_dim, 4: N-cluster
    :return:
    """

    # Step 1: Read input dataset
    # ---------------------------------
    data1 = pd.read_csv(argv[0])

    # taking a sample of 10 documents
    data = data1[:10]

    # Step 2: Pre-processing the data
    # ----------------------------------
    pre_process_columns = TextCleaning()
    data['AbstractClean'] = data['Abstract'].apply(pre_process_columns.normalize_corpus,
                                                       args=(argv[0], True, True, True, True, True, True, argv[2]))
    data['PaperTextClean'] = data['PaperText'].apply(pre_process_columns.normalize_corpus,
                                                         args=(argv[0], True, True, True, True, True, True, argv[2]))
    # Backup cleaned data
    data.to_csv("./input/cleaned_data.csv")

    # Step 3: Word embeddings - Fast-text
    # -------------------------------------
    word_embedding = WordEmbedding(feature_size=argv[3])
    word_embedding.build_model(data)

    # find word similarity for list of words
    words_similarity = ["classification","experiments"]
    print(word_embedding.get_similar_words(words_similarity))

    # Step 4: Generate keywords for each doc
    # ---------------------------------------
    get_keywords = GenerateAspects(data)
    get_keywords.gen_aspect()

    # Step 5: Generate document cluster
    # ---------------------------------------
    gen_cluster = DocumentClustering(data)
    gen_cluster.get_document_cluster(topics_cnt=5)


if __name__ == "__main__":
    main(sys.argv[1:])
