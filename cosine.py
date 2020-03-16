from docx import Document
import pandas as pd
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
tfidf_vectorizer = TfidfVectorizer()


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
    return tokens


def preprocess(df, stem=False, lemma=False):
    tokens = []
    if stem:
        tokens = df.sent.apply(lambda x: tokenize(x, True))
    if lemma:
        tokens = df.sent.apply(lambda x: tokenize(x, False, True))
    return tokens


def cosine(tokens):
    # print(tokens[1])
    # print(tokens.count())
    # for count in tokens.count():
        tokens = tokens.apply(lambda x: ' '.join(x))
        sent1 = tokens[9]
        sent2 = tokens[10]
        print(sent1)
        print('-----------------------------')
        print(sent2)
        documents = [sent1, sent2]

        print('count vectorize')
        count_vectorizer = CountVectorizer(stop_words='english')
        sparse_matrix = count_vectorizer.fit_transform(documents)
        print(cosine_similarity(sparse_matrix[0:1], sparse_matrix))

        print('tfidf')
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        # doc_term_matrix = sparse_matrix.todense()
        print(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix))

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
    df_doc = read_files('./data/Report 1_Industry4.0.docx')
    tokens = preprocess(df_doc, stem=stem, lemma=lemma)
    cosine(tokens)


def main():
    print('with stemming')
    run(stem=True, lemma=False)
    print('with lemmatizing')
    run(stem=False, lemma=True)


if __name__ == '__main__':
    main()
