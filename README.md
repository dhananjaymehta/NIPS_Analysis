# NIPS Data Analysis
## Get the data
We will be using the Neural Information Processing Systems (NIPS) 2015 conference papers as a text corpus. These can be obtained off Kaggle, which requires that you create a free account, at [this link](https://www.kaggle.com/benhamner/nips-2015-papers/version/2). An [overview](https://www.kaggle.com/benhamner/nips-2015-papers/version/2/home) talks about the data, which is available as a csv, in raw form from the PDFs, and also already stored into a sqlite db.

## Programming Language
All work should be completed in Python. **Please use Python 3.6+!!**

## Basic Info
Although we are only going to be working with the NIPS dataset, consider that this might be used later on to process a dataset much larger (eg an entire subreddit or all the articles from a news publication). Your ideas should be reusable and scaleable. Final work should be in commandline executable form and should allow the user to start with the files mentioned above and do all the following tasks.

## Tasks
### Generate FastText word embeddings
One of the standard building blocks in NLP is the word embedding. The gold standard used to be word2vec embeddings, however we will be creating FastText embeddings instead due to their ability to deal with out-of-vocabulary words. Ensure that you can support a user-defined number of dimensions and have the ability to turn on/off at least one text-preprocessing step.

### Generate a list of keywords
Within every text corpus, there are certain keywords that are specific to that corpus. Although keywords may normally be monograms, be sure that your methods can identify at least bigrams as well.

### Group the documents
In addition to keywords, a corpus will generally have clusters of topics. Group the documents together (ie output groups of document titles) and identify how they are grouped.

## Additional Info
Logical organization and comments help to make code more understandable and maintainable. Keep this in mind as you write your code! Feel free to use any packages that you would like. Please check all your code into a publicly accessible Github repo and send me an email with link to your work.


### NOTE: Major Consideration
- Design Decisions
- Code pattern
- Documentation
- Testing - Using Visualization


#### design consideration - 
1. can extend the cleaning for html data
2. Can save model and reuse it - can perform various functions such as  - closeness of words, antonyms, grouping
3. Keywords identified in task 2 - these can be used to see related words in corpus, we can give these inputs to see related words from vectorizer
4. Clustering can use the words for generating the categories for document classification, in task 3 
5. Based on the common features, we can use the tags to identify related words: The Tags can be grouped under similar words - and these can be used to identify Group similar words
 
 
 ## Components:
 
Text Mining (i.e. Text clustering, data-driven topics)
Categorization (i.e. Tagging unstructured data into categories and sub-categories; hierarchies; taxonomies)
Entity Extraction (i.e. Extracting patterns such as phrases, addresses, product codes, phone numbers, etc.)
Sentiment Analysis (i.e. Tagging positive, negative, or neutral with varying levels of sentiment)
Deep Linguistics (i.e Semantics. Understanding causality, purpose, time, etc.)

### Task1
There are following steps that are taken for generating word embedding - 

Input: Pre-processed data for Abstract/Text column

Step 1: Tokenize the words,
The input sentences are being tokenized, I am taking a backup of the words to make sure we have a bag of words that can be later used for  **entity extraction**,  

Step 2: Build model

Step 3: call model to generate words
 
Output: List of similar words


### Task 2
I want to generate aspects based on either text or abstract, I want to give the flexibility of any sub component can be used for doing the processing
note that currently the document is huge but if this big document was to be replaced with a group of texts we can aggregate the sentences together to one giant doc

I know pandas is not great but using it to show how we can run these transformations with Spark DF <https://www.dataquest.io/blog/pandas-big-data/>