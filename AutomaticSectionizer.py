#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import docx


# In[2]:


from sklearn.feature_extraction import text


# In[3]:


doc = docx.Document("report19.docx")       # open connection to Word Document
result = [p.text for p in doc.paragraphs]  # read in each paragraph in file


# In[4]:


nltk.download('wordnet')


# In[5]:


df = pd.DataFrame({'text':result, 'para#': range(1,len(result) + 1)}) #dataframe with column headers assigned
df.index += 1 #set index.. reset later.


# In[6]:


df.text[1]

######adding stopwords usign union "text.ENGLISH_STOP_WORDS.union(["---words---"])"
# In[7]:


stops = text.ENGLISH_STOP_WORDS


# In[8]:


df['text'] = df['text'].str.lower()
df['text_token'] = df.text.apply(lambda x: word_tokenize(x))


# In[9]:


df.head()


# In[10]:


def clean(text):
    cleaned = [w for w in text if w not in stops]
    clean = [w for w in cleaned if w not in string.punctuation]
    return ' '.join(clean)

#### Insert Regular Expressions
####
####


# In[11]:


df['clean_text'] = df['text_token'].apply(clean)


# In[12]:


lemmatizer = WordNetLemmatizer()

def lemmat(wor):
    l = []
    for i in wor:
        l.append(lemmatizer.lemmatize(i))
    return l


# In[13]:


df['stem_token'] = df.clean_text.apply(lambda x: word_tokenize(x))
df['stem_token'] = df['stem_token'].apply(lemmat)


# In[14]:


df['clean_text'] = df['stem_token'].apply(lambda x: ' '.join(x))


# In[15]:


df.head()


# In[16]:


stem_df = df.clean_text.values.tolist()


# In[17]:


stem_df = pd.DataFrame({'text':stem_df})


# In[18]:


stem_df.head()


# In[19]:


print (df.text[1])
print ('-'*100)
print (df.clean_text[1])
print ('-'*100)
print (df.text_token[1])
print ('-'*100)
print (df.stem_token[1])


# In[20]:


from sklearn.feature_extraction.text import CountVectorizer


# In[21]:


cv = CountVectorizer(min_df=1)
X = cv.fit_transform(stem_df.text)


# In[22]:


data = pd.DataFrame({'features':cv.get_feature_names(),'occurance':X.toarray().sum(axis=0)})
data = data.sort_values('occurance',ascending=False)
data.head(30)


# In[23]:


features = pd.DataFrame(X.toarray(),columns=cv.get_feature_names())


# In[24]:


features.shape


# In[ ]:




