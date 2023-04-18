import numpy as np 
import pandas as pd
import nltk
nltk.download('all')
from nltk.corpus import stopwords
#from nltk. corpus import PorterStemmer
import itertools
import matplotlib.pyplot as plt 
import scipy.stats as stats
import seaborn as sns 
sns.set_style("darkgrid")
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import fasttext
from pycountry import languages

from wordcloud import WordCloud, STOPWORDS
import matplotlib.cm as cm
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

data= pd.read_excel('/Users/prathamkaveriappa/Desktop/demo_input.xlsx')
data.head(3)
data.shape
data.columns

data["Assignment group"].unique()
data[data.isnull().any(axis=1)]
data.dropna(inplace=True)
data.shape
duplicate_data=data[['Short description', 'Description', 'Caller','Assignment group']].copy()
duplicateRowsDF = duplicate_data[duplicate_data.duplicated()]
duplicateRowsDF
updated_data=data.drop_duplicates(['Short description', 'Description', 'Caller', 'Assignment group'])
data_grp = updated_data.groupby(['Assignment group']).size().reset_index(name='counts')
data_grp
data_grp.describe()
data_sum = pd.DataFrame({"Description": updated_data["Short description"] + " " + updated_data["Description"],"AssignmentGroup": updated_data["Assignment group"]}, columns=["Description","AssignmentGroup"])
#downloaidn nltk packages
nltk.download('stopwords')
stop = set(stopwords.words('english')) 
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
#cleaning data 
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
lemmatizer = WordNetLemmatizer()
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)

temp =[]
for sentence in data_sum["Description"]:
    sentence = sentence.lower()
    #sentence = sentence.str.replace('\d+', '')
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, ' ', sentence)        #Removing HTML tags
    sentence = re.sub(r'\S+@\S+', 'EmailId', sentence)
    sentence = re.sub(r'\'', '', sentence, re.I|re.A)
    sentence = re.sub(r'[0-9]', '', sentence, re.I|re.A)
    #print ("Sentence1.5 = ",sentence)
    sentence = re.sub(r'[^a-zA-Z0-9\s]', ' ', sentence)
    #print ("Sentence2 = ",sentence)
    sentence = sentence.lower()
    sentence = re.sub(r'com ', ' ', sentence, re.I|re.A)
    sentence = re.sub(r'hello ', ' ', sentence, re.I|re.A)
    l_sentence = lemmatize_sentence(sentence)

    words = [word for word in l_sentence.split() if word not in stopwords.words('english')]
    temp.append(words)
data_sum["Lemmatized_clean"] = temp

import fasttext
from pycountry import languages
PRETRAINED_MODEL_PATH = '/tmp/lid.176.bin'
model = fasttext.load_model(PRETRAINED_MODEL_PATH)

temp1 =[]
temp2 = []
for sentence in data_sum["Lemmatized_clean"]:
    acc = 0
    try:
      predictions = model.predict(sentence)
      prediction_lang = re.sub('__label__', '',str(predictions[0][0][0]))
      acc = round(predictions[1][0][0],2) * 100
      language_name = languages.get(alpha_2=prediction_lang).name
    except:
      language_name = "NOT DETERMINED"
    temp1.append(language_name)
    temp2.append(acc)

#vectorization

