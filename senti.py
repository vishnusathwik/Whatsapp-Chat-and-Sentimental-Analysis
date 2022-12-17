
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from nltk.corpus import sentiwordnet as swn
import numpy as np
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import emoji
import itertools
from collections import Counter
import warnings
import os
warnings.filterwarnings('ignore')
from analysis import *
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
# Define a function to clean the text
def clean(text):
    text = re.sub('[^A-Za-z]+', ' ', str(text)) 
    return str(text)

# function to calculate subjectivity 
def sentimental(filename):
    df = pd.read_csv(os.path.join('csvs',filename))
    df['Cleaned Reviews'] = df['Messages'].apply(clean)
    df['Lemma'] = df['POS tagged'].apply(lemmatize)
    fin_data = pd.DataFrame(df[['Cleaned Reviews', 'Lemma']])
    fin_data['SWN analysis'] = df['POS tagged'].apply(sentiwordnetanalysis)
    swn_counts= fin_data['SWN analysis'].value_counts()
    plt.figure(figsize=(10, 7))
    plt.pie(swn_counts.values, labels = swn_counts.index, autopct='%1.1f%%', shadow=False)
    plt.title("Sentimental analysis", fontsize=20)
    sns.despine( left=True, bottom=True)
    plt.savefig(os.path.join('static/images/dashboard',filename+'sentipie.png') ,bbox_inches='tight')

def sentiwordnetanalysis(pos_data):
    sentiment = 0
    tokens_count = 0
    for word, pos in pos_data:
        if not pos:
            continue
        lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
        if not lemma:
            continue
        synsets = wordnet.synsets(lemma, pos=pos)
        if not synsets:
            continue
        synset = synsets[0]
        swn_synset = swn.senti_synset(synset.name())
        sentiment += swn_synset.pos_score() - swn_synset.neg_score()
        tokens_count += 1
    if not tokens_count:
        return "Not Identified"
    if sentiment>0:
        return "Positive"
    if sentiment==0:
        return "Neutral"
    else:
        return "Negative"
def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist

def lemmatize(pos_data):
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos: 
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:  
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew

def token_stop_pos(text):
    tags = pos_tag(word_tokenize(text))
    newlist = []
    for word, tag in tags:
        if word.lower() not in set(stopwords.words('english')):
            newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist
