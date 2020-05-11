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
from gensim.matutils import softcossim
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess
print(gensim.__version__)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()



def read_files(path):
    document = Document(path)
    doc = []
    for para in document.paragraphs:
        if len(para.text) > 201:
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


def cosine(tokens, stem=False, lemma=False):
    cosout_tfidf = []
    cosout_cv = []
    colnames = []
    df_cosine_cv = pd.DataFrame()
    df_cosine_tfidf = pd.DataFrame()
    df_cosine = pd.DataFrame()

    tokens = tokens.apply(lambda x: ' '.join(x))
    for count in range(0, len(tokens) - 1):
            sent1 = tokens[count]
            sent2 = tokens[count + 1]
            # print(count + 1, count + 2)
            print(sent1)
            print('-----------------------------')
            print(sent2)
            documents = [sent1, sent2]
            parag1 = 'parag#' + str(count + 1)
            parag2 = ' & ' + str(count + 2)
            paragnumber = parag1 + parag2

            print('count vectorize')
            count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3))
            sparse_matrix = count_vectorizer.fit_transform(documents)
            # print(cv)
            cv = cosine_similarity(sparse_matrix[0:1], sparse_matrix)
            print(cv)
            cosout_cv.append(cv[0][1])
            print('tfidf')
            # tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
            tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
            tfidf = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
            cosout_tfidf.append(tfidf[0][1])
            print(tfidf)
            colnames.append(paragnumber)
    df_cosine = pd.DataFrame(list(zip(cosout_cv, cosout_tfidf)),
                      columns=['cosout_cv', 'cosout_tfidf'])
    print(df_cosine)
    if stem:
        df_cosine.to_csv(r'C:/Users/prash/OneDrive/Documents/Prashanti/DAEN 690/basiccosine_stem_report11.csv',
                                     index=None, header=True)
    if lemma:
        df_cosine.to_csv(r'C:/Users/prash/OneDrive/Documents/Prashanti/DAEN 690/basiccosine_lemma_report11.csv',
                                       index=None, header=True)

    # print(df_cosine)


            # cosine.update(count + 1, count + 2, cv, tfidf)
        # cosine_df = pd.DataFrame(cosine_df)
    # print(cosine_df)
    # return cosine_df
        # tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        # print('hash')
        # hashing = HashingVectorizer(stop_words='english', ngram_range=(1, 3))
        # hashing_matrix = hashing.fit_transform(documents)
        # print(cosine_similarity(hashing_matrix[0:1], hashing_matrix))


        # doc_term_matrix = sparse_matrix.todense()
        # print(cosine_similarity(df, df))

    # values = tfidf_vectorizer.fit_transform(tokens)
    #
    # # Show the Model as a pandas DataFrame
    # feature_names = tfidf_vectorizer.get_feature_names()
    # result = pd.DataFrame(values.toarray(), columns=feature_names)
    #
    # print(result)
    #
    #
    # print(cosine_similarity(result, result))


def run(stem=False, lemma=False):
    df_doc = read_files('./data/Report 11_Automotive.docx')
    tokens = preprocess(df_doc, stem=stem, lemma=lemma)
    cosine(tokens, stem=stem, lemma=lemma)


def main():
    print('with stemming')
    run(stem=True, lemma=False)
    print('with lemmatizing')
    run(stem=False, lemma=True)


if __name__ == '__main__':
    main()
