import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import docx
from sklearn.feature_extraction import text
from nltk.stem.porter import PorterStemmer

# set display width
desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 10)

doc = docx.Document("./data/Report 19_India Economy.docx")       # open connection to Word Document
result = [p.text for p in doc.paragraphs]  # read in each paragraph in file

df = pd.DataFrame({'text':result, 'para#': range(1,len(result) + 1)}) #dataframe with column headers assigned
df.index += 1 #set index.. reset later.


#convert to lowercase
df['text'] = df['text'].str.lower()
#create work tokens
df['text_token'] = df.text.apply(lambda x: word_tokenize(x))

#remove stop words and punctuations
stops = text.ENGLISH_STOP_WORDS
def clean(text):
    cleaned = [w for w in text if w not in stops]
    clean_token = [w for w in cleaned if w not in string.punctuation]
    return clean_token

df['clean_token'] = df['text_token'].apply(clean)

#lemmatize
lemmatizer = WordNetLemmatizer()

def lemmat(wor):
    l = []
    for i in wor:
        l.append(lemmatizer.lemmatize(i))
    return l

df['lemma_token'] = df['clean_token'].apply(lemmat)

#stemming
stemmer = PorterStemmer()

def stem(wor):
    l = []
    for i in wor:
        l.append(stemmer.stem(i))
    return l

df['stem_token'] = df['clean_token'].apply(stem)

print(df.head())