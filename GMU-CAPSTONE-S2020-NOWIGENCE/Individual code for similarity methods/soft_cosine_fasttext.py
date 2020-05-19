from docx import Document
import pandas as pd
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
from gensim.matutils import softcossim
from gensim import corpora
import gensim.downloader as api
from gensim.utils import simple_preprocess
# load fasttext embeddings
fasttext_model300 = api.load('fasttext-wiki-news-subwords-300')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
#

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
    tokens = [token for token in tokens if token not in stopwords.words('english')]
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

#
def soft_cosine(tokens, stem=False, lemma=False):
    """
    Apply soft cosine between two paragraphs using fasttext embeddings and append the scores to a dataframe
    :param tokens:
    :param stem: if stem is true, output cosine scores are saved for paragraphs with stemmed tokens
    :param lemma: if lemma is true, output cosine scores are saved for paragraphs with lemmatized tokens
    :return: none
    """
    softcosout = []
    colnames = []
    df_softcos = pd.DataFrame()
    tokens = tokens.apply(lambda x: ' '.join(x))
    
    token_1 = []
    token_2 = []
    
    for count in range(0, len(tokens)-1):
        sent1 = tokens[count]
        sent2 = tokens[count+1]

        parag1 = 'parag#' + str(count+1)
        parag2 = ' & ' + str(count+2)
        paragnumber = parag1 + parag2
        parag_1 = str(count+1)
        parag_2 = str(count+2)

        documents = [sent1, sent2]
        # create vocabulary
        dictionary = corpora.Dictionary([simple_preprocess(doc) for doc in documents])
        # apply fasttext model
        similarity_matrix = fasttext_model300.similarity_matrix(dictionary, tfidf=None, threshold=0.0, exponent=2.0,
                                                                nonzero_limit=100)
        # create bag of words
        sent_1 = dictionary.doc2bow(simple_preprocess(sent1))
        sent_2 = dictionary.doc2bow(simple_preprocess(sent2))
        # apply softcosine similarity
        soft_cosine_output = softcossim(sent_1, sent_2, similarity_matrix)
        print(soft_cosine_output)
        colnames.append(paragnumber)
        softcosout.append(soft_cosine_output)
        token_1.append(parag_1)
        token_2.append(parag_2)
    # create and export dataframe
    df_softcos = pd.DataFrame(softcosout, columns = ['Soft cosine'])
    df_softcos['token 1'] = token_1
    df_softcos['token 2'] = token_2
    print(df_softcos)
    
    if stem:
        df_softcos.to_csv(r'softcosine_stem_output15.csv',
                      index=None, header=True)
    if lemma:
        df_softcos.to_csv(r'softcosine_lemma_output15.csv',
                              index=None, header=True)



def run(stem=False, lemma=False):
    """
        run - Execution of appropriate functions as per the required call
    """
    df_doc = read_files('Report 15_IIOT.docx')
    tokens = preprocess(df_doc, stem=stem, lemma=lemma)
    soft_cosine(tokens, stem=stem, lemma=lemma)


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
