from docx import Document
import pandas as pd
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def read_files(path):
    """ reads one of the 19 reports in docx format based on the given path
    considers paragraphs greater than 201 characters
    creates a dataframe with rows consisting of paragraphs
    :param path:
    :return: dataframa
    """
    document = Document(path)
    doc = []
    for para in document.paragraphs:
        if len(para.text) > 201:
            doc.append(para.text)
    df = pd.DataFrame(doc, columns=['sent'])
    return df


def tokenize(x, stem=False, lemma=False):
    """ html tags, numbers, and special characters in each paragraph are replaced with space and sentences are tokenized.
        Apply stemming/lemmatization and return tokens.
        :param x: paragraph
        :param stem: apply stemming if stem is true
        :param lemma: apply lemmatization of lemma is true
        :return: tokens
    """
    x = re.sub(r'(?:<[^>]+>)', '', x)
    pattern = '0-9 $ ` % {} <> -,: _ \ . = +| /'
    x = re.sub(r'(pattern)', '', x)
    tokens = word_tokenize(x)
    if stem:
        tokens = [stemmer.stem(t) for t in tokens]
    if lemma:
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def preprocess(df, stem=False, lemma=False):
    """
        :param df: input dataframe containing paragraphs which are preprocessed based on stem and lemma parameters.
        :param stem: call tokenize function and apply stemming if stem is true
        :param lemma: call tokenize function and apply lemmatization of lemma is true
        :return:
    """
    tokens = []
    if stem:
        tokens = df.sent.apply(lambda x: tokenize(x, True))
    if lemma:
        tokens = df.sent.apply(lambda x: tokenize(x, False, True))
    return tokens


def cosine(tokens, stem=False, lemma=False):
    """
       Apply cosine similarity between two paragraphs and append the scores to a dataframe
       :param tokens:
       :param stem: if stem is true, output cosine scores are saved for paragraphs with stemmed tokens
       :param lemma: if lemma is true, output cosine scores are saved for paragraphs with lemmatized tokens
       :return: none
       """
    cosout_tfidf = []
    cosout_cv = []
    token_names = []
    df_cosine_cv = pd.DataFrame()
    df_cosine_tfidf = pd.DataFrame()
    df_cosine = pd.DataFrame()
    
    token_1 = []
    token_2 = []

    tokens = tokens.apply(lambda x: ' '.join(x))
    for count in range(0, len(tokens) - 1):
            sent1 = tokens[count]
            sent2 = tokens[count + 1]

            documents = [sent1, sent2]
            parag1 = 'parag#' + str(count + 1)
            parag2 = ' & ' + str(count + 2)
            paragnumber = parag1 + parag2
            
            parag_1 = str(count+1)
            parag_2 = str(count+2)

            # create bag of words using count vectorizer with ngram range 1-3
            count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 3))
            sparse_matrix = count_vectorizer.fit_transform(documents)

            cv = cosine_similarity(sparse_matrix[0:1], sparse_matrix)
            cosout_cv.append(cv[0][1])
            # create bag of words using tf-idf vectorizer with ngram range 1-3
            tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3))
            tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
            tfidf = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
            cosout_tfidf.append(tfidf[0][1])
            token_names.append(paragnumber)
            token_1.append(parag_1)
            token_2.append(parag_2)
    # create and export dataframe
    df_cosine = pd.DataFrame(list(zip(cosout_cv, cosout_tfidf)),
                      columns=['cosout_cv', 'cosout_tfidf'])
    df_cosine['token 1'] = token_1
    df_cosine['token 2'] = token_2
    
    if stem:
        df_cosine.to_csv(r'basiccosine_stem_output11.csv',
                                     index=True, header=True)
    if lemma:
        df_cosine.to_csv(r'basiccosine_lemma_output11.csv',
                                       index=True, header=True)



def run(stem=False, lemma=False):
    """
         run - Execution of appropriate functions as per the required call
    """
    df_doc = read_files('Report 11_Automotive.docx')
    tokens = preprocess(df_doc, stem=stem, lemma=lemma)
    cosine(tokens, stem=stem, lemma=lemma)


def main():
    """
           main - runs all the modules via run function
    """
    print('with stemming')
    run(stem=True, lemma=False)
    print('with lemmatizing')
    run(stem=False, lemma=True)


if __name__ == '__main__':
    main()
