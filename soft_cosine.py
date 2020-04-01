from docx import Document
import pandas as pd
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

import gensim
# upgrade gensim if you can't import softcossim
from gensim.matutils import softcossim
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess
fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
#
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
#


def read_files(path):
    document = Document(path)
    doc = []
    for para in document.paragraphs:
        if len(para.text) > 101:
            doc.append(para.text)
    df = pd.DataFrame(doc, columns=['sent'])
    return df


def tokenize(x, stem=False, lemma=False):
    x = re.sub(r'(?:<[^>]+>)', '', x)
    pattern = '0-9 $ ` % {} <> -,: _ \ . = +| /'
    x = re.sub(r'(pattern)', '', x)
    tokens = word_tokenize(x)
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    if stem:
        tokens = [stemmer.stem(t) for t in tokens]
    if lemma:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    # sent_bigrams = tokens.apply(lambda tweet: [' '.join(bigram) for bigram in ngrams(tweet, 2)])
    return tokens


def preprocess(df, stem=False, lemma=False):
    tokens = []
    if stem:
        tokens = df.sent.apply(lambda x: tokenize(x, True))
    if lemma:
        tokens = df.sent.apply(lambda x: tokenize(x, False, True))
    return tokens

#
def soft_cosine(tokens, stem=False, lemma=False):
    softcosout = []
    colnames = []

    df_softcos = pd.DataFrame()
    tokens = tokens.apply(lambda x: ' '.join(x))
    for count in range(0, len(tokens)-1):
        sent1 = tokens[count]
        sent2 = tokens[count+1]
        print(count+1, count+2)
        parag1 = 'parag#' + str(count+1)
        parag2 = ' & ' + str(count+2)
        paragnumber = parag1 + parag2
        # print(sent1)
        # print('-----------------------------')
        # print(sent2)
        documents = [sent1, sent2]
        dictionary = corpora.Dictionary([simple_preprocess(doc) for doc in documents])
        similarity_matrix = fasttext_model300.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0,
                                                                nonzero_limit=100)
        sent_1 = dictionary.doc2bow(simple_preprocess(sent1))
        sent_2 = dictionary.doc2bow(simple_preprocess(sent2))
        soft_cosine_output = softcossim(sent_1, sent_2, similarity_matrix)
        print(soft_cosine_output)
        softcosout.append(soft_cosine_output)
        colnames.append(paragnumber)
    df_softcos = pd.DataFrame(softcosout, index=colnames)
    print(df_softcos)
    if stem:
        df_softcos.to_csv(r'C:/Users/prash/OneDrive/Documents/Prashanti/DAEN 690/cosine_stem_output.csv',
                                     index=None, header=True)
    else:
        df_softcos.to_csv(r'C:/Users/prash/OneDrive/Documents/Prashanti/DAEN 690/cosine_lemma_output.csv',
                                       index=None, header=True)
    return df_softcos


def run(stem=False, lemma=False):
    df_doc = read_files('./data/Report 1_Industry4.0.docx')
    tokens = preprocess(df_doc, stem=stem, lemma=lemma)
    df_softcos = soft_cosine(tokens, stem=stem, lemma=lemma)


def main():
    print('with stemming')
    run(stem=True, lemma=False)
    print('with lemmatizing')
    run(stem=False, lemma=True)


if __name__ == '__main__':
    main()
