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

from gensim import models
# upgrade gensim if you can't import softcossim
from gensim.matutils import softcossim
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess
# fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')
# wiki = api.load("wiki-en") #word embedding

from gensim.models.word2vec import Word2Vec
# model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# model = models.KeyedVectors.load_word2vec_format(
#     './GoogleNews-vectors-negative300.bin', binary=True)
#
# model = api.load('word2vec-google-news-300')
model = api.load('glove-wiki-gigaword-300')
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
#


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
    # softcosout_stem = []
    # softcosout_lemma = []
    colnames = []
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
        similarity_matrix = model.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0,
                                                                nonzero_limit=100)
        sent_1 = dictionary.doc2bow(simple_preprocess(sent1))
        sent_2 = dictionary.doc2bow(simple_preprocess(sent2))

        soft_cosine_output = softcossim(sent_1, sent_2, similarity_matrix)
        print(soft_cosine_output)

        softcosout.append(soft_cosine_output)
        colnames.append(paragnumber)
        # if lemma:
        #     softcosout_lemma.append(soft_cosine_output)
    df_softcos = pd.DataFrame(softcosout, index=colnames)
    print(df_softcos)
    if stem:
        df_softcos.to_csv(r'C:/Users/prash/OneDrive/Documents/Prashanti/DAEN 690/glove_stem_repo5.csv',
                                     index=None, header=True)
    else:
        df_softcos.to_csv(r'C:/Users/prash/OneDrive/Documents/Prashanti/DAEN 690/glove_lemma_repo5.csv',
                                       index=None, header=True)



def run(stem=False, lemma=False):
    df_doc = read_files('./data/Report 5_FMCG.docx')
    tokens = preprocess(df_doc, stem=stem, lemma=lemma)
    soft_cosine(tokens, stem=stem, lemma=lemma)



def main():

    print('with stemming')
    # softcosout_stem, _ = run(stem=True, lemma=False)
    run(stem=True, lemma=False)
    print('with lemmatizing')
    # _, softcosout_lemma = run(stem=False, lemma=True)
    run(stem=False, lemma=True)
    # df_softcos = pd.DataFrame(softcosout_stem, softcosout_lemma)
    # print(df_softcos)

if __name__ == '__main__':
    main()
