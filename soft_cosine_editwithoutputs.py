#DAEN 690, Spring 2020
#March 25, 2020


#to update gensim
#pip install gensim
#pip install --upgrade gensim

from docx import Document
import pandas as pd
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

import gensim
# upgrade gensim if you can't import softcossim
from gensim.models import FastText
from gensim.matutils import softcossim
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess
fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()



def read_files(path):
    document = Document(path)
    doc = []
    for para in document.paragraphs:
        doc.append(para.text)
    df = pd.DataFrame(doc, columns=['sent'])
    return df


def tokenize(x, stem=False, lemma=False):
    x = re.sub(r'(?:<[^>]+>)', '', x)
    pattern = '0-9 $ ` % {} <> -,: _ \ . = +| /'
    x = re.sub(r'(pattern)', '', x)
    tokens = word_tokenize(x)
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



def soft_cosine(tokens):
    softcosout = []
    colnames = []
    
    df_softcos = pd.DataFrame()

    tokens = tokens.apply(lambda x: ' '.join(x))
    for count in range(0, len(tokens)-1):
        sent1 = tokens[count]
        sent2 = tokens[count+1]
        
        parag1 = 'parag#' + str(count)
        parag2 = ' & ' + str(count+1)
        paragnumber = parag1+parag2
        
        print("parag #", count, 'parag#', count+1)
        print(sent1)
        print('-----------------------------')
        print(sent2)
        documents = [sent1, sent2]
        dictionary = corpora.Dictionary([simple_preprocess(doc) for doc in documents])
        similarity_matrix = fasttext_model300.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0,
                                                                nonzero_limit=100)
        sent_1 = dictionary.doc2bow(simple_preprocess(sent1))
        sent_2 = dictionary.doc2bow(simple_preprocess(sent2))
        
        #print(softcossim(sent_1, sent_2, similarity_matrix))

        softcosoutput = softcossim(sent_1, sent_2, similarity_matrix)
        print(softcosoutput)
        #print(type(softcosoutput))
        
        softcosout.append(softcosoutput)
        colnames.append(paragnumber)
        #print(gensim.similarities.termsim.SparseTermSimilarityMatrix.inner_product
    
    print(softcosout)
    print(colnames)
    df_softcos = pd.DataFrame(softcosout, index = colnames)
    print(df_softcos)
    



def run(stem=False, lemma=False):
    df_doc = read_files('Report 2_Telecommunications.docx')
    tokens = preprocess(df_doc, stem=stem, lemma=lemma)
    soft_cosine(tokens)


def main():
    print('with stemming')
    run(stem=True, lemma=False)
    print('with lemmatizing')
    run(stem=False, lemma=True)


if __name__ == '__main__':
    main()
