import pandas as pd
from text_preprocessing import TextCleaning
from task1_word_embedding import WordEmbedding
from task2_aspects import GenerateAspects
from task3_document_clustering import DocumentClustering

def main():
    """
    Step1:
    Step2....
    :return:
    """

    # Step1: Read input dataset
    # ---------------------------------
    df = pd.read_csv("./data/Papers.csv")

    # taking a sample of 10 documents
    sub_df = df[:10]
    sub_df.to_csv("./data/Papers_sub.csv")
    """
    # Step2: Pre-processing the data
    # ----------------------------------
    pre_process_columns = TextCleaning()

    # Add cleaned text column
    sub_df['AbstractClean'] = sub_df['Abstract'].apply(pre_process_columns.normalize_corpus,
                                                       args=(False, True, True, True, True, True, True, True))

    sub_df['PaperTextClean'] = sub_df['PaperText'].apply(pre_process_columns.normalize_corpus,
                                                         args=(False, True, True, True, True, True, True, True))

    sub_df.to_csv("./data/cleaned_data.csv")

    # Step 3: Generate word vectors - using Fast-text
    # --------------------------------------------------
    obj = WordEmbedding()

    # build model for dataset
    obj.build_model(sub_df)

    # get word similar
    words_similarity = ["classification"]
    print(obj.get_similar_words(words_similarity))

    # Step 3: Get Aspects
    get_keywords = GenerateAspects(sub_df)
    get_keywords.gen_aspect()

    #df = pd.read_csv("./data/cleaned_data.csv")

    # Step 4: Get The documents cluster
    gen_cluster=DocumentClustering(sub_df)
    gen_cluster.get_document_cluster(topics_cnt=5)
    """


if __name__ == "__main__":
    main()