from flask import Flask, request, render_template
from datetime import datetime
import plotly.express as px
import collections
import seaborn as sb
import string
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from senti import *
import pandas as pd
import datetime
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
from nltk import pos_tag
from nltk.corpus import wordnet
pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def pie(filename):
    df = pd.read_csv(os.path.join('csvs',filename))
    plt.figure(figsize=(4,4))
    recipe = list( df.groupby('Shift').count()['Time'].index )
    data = list(df.groupby('Shift').count()['Time'].values)
    lable = list([str(recipe[0] + '\n'+str(data[0])+' msgs') ,str(recipe[1] + '\n'+str(data[1])+' msgs')])
    fig = px.pie(values=data, names=lable, title='Messages in respective Meridian')
    fig.write_html(os.path.join('static/',filename+'pie.html'))
'''    plt.pie(data, textprops=dict( fontsize=15,
        color="black"), wedgeprops=dict(width=0.45), startangle=20 ,labels=lable)

    plt.title("Messages in respective Meridian", fontsize=20)
    sb.despine( left=True, bottom=True)
    plt.savefig(os.path.join('static/images/dashboard',filename+'pie.png') ,bbox_inches='tight')'''


def word(filename):
    df = pd.read_csv(os.path.join('csvs',filename))
    plt.figure(figsize=(8,12))
    new_stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your',
                'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
                "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 
                'that', "that'll", 
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 
                'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 
                'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 
                'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 
                'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 
                'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 
                'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
                'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn',
                "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",'1','2','3','4','5','6','7',
                '8','9','0','.',',','/','!','@','#','$','%','^','&','*','(',')','+','-','media','omitted','media omitted','nan','message deleted'
               ]

    for stop in new_stop:
        STOPWORDS.add(stop)

    i = 0

    comment_words = ' '
    stopwords = set(STOPWORDS) 

    # iterate through the csv file 
    for val in df['Messages']: 

        # typecaste each val to string 
        val = str(val) 

        if "media omitted" in val:
            i+=1
        # split the value 
        tokens = val.split() 
        # Converts each token into lowercase 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 

        for words in tokens: 
            comment_words = comment_words + words + ' '


    wordcloud = WordCloud(width = 700, height = 500,
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10,
                    max_font_size = 150,      
                    colormap= 'tab20').generate(comment_words) 

    plt.title("WORD CLOUD",fontsize=40)
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.savefig(os.path.join('static/images/dashboard',filename+'word.svg') ,bbox_inches='tight')

# return the number of messages in the csv file
def number_of_msgs(filename):
    df = pd.read_csv(os.path.join("csvs/", filename))
    return df.shape[0]


# return the number of unique members of the group
def number_of_unique_members(filename):
    df = pd.read_csv(os.path.join("csvs/", filename))
    l = df["Contacts"].unique()
    count=0
    for i in l:
        if(i.isnumeric()):
            count+=1
            
    return (len(l)-count)


# return the starting date of the group
def start_date(filename):
    df = pd.read_csv(os.path.join("csvs", filename))
    return df["Date"][0]


# return the end date of the group
def end_date(filename):
    df = pd.read_csv(os.path.join("csvs", filename))
    return df["Date"][df.shape[0] - 1]


def average_length_msg(filename):
    df = pd.read_csv(os.path.join("csvs/", filename))
    i = 0
    for msg in df["Messages"]:
        i += len(str(msg).split(" "))
    return str(i / df.shape[0])


def max_length_msg(filename):
    df = pd.read_csv(os.path.join("csvs/", filename))
    i = 0
    name = ""
    for msg in df["Messages"]:
        if i < len(str(msg).split(" ")):
            if df[df["Messages"] == msg]["Contacts"].shape[0] > 0:
                i = len(str(msg).split(" "))
                name = df[df["Messages"] == msg]["Contacts"].values[0]
    return (i, name)


def weekday_busy(filename):
    df = pd.read_csv(os.path.join("csvs/", filename))
    week = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
    return week[Counter(pd.to_datetime(df["Date"]).dt.weekday).most_common(1)[0][0]]


def month_busy(filename):
    df = pd.read_csv(os.path.join("csvs/", filename))
    month = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December",
    }
    return month[Counter(pd.to_datetime(df["Date"]).dt.month).most_common(1)[0][0]]


def most(filename):
    df = pd.read_csv(os.path.join("csvs", filename))
    plt.figure(figsize=(8, 6))
    sorted_active = df.groupby("Contacts").count()["Time"].sort_values()
    fig = px.bar(sorted_active, x=sorted_active[-10:].values, y=sorted_active[-10:].index, title='Most Active Users')
    fig.update_layout(xaxis_title='Messages', yaxis_title='Users')
    fig.write_html(os.path.join('static/',filename+'mactive.html'))
'''    if df.groupby("Contacts").count().shape[0] > 10:
        fig = px.bar(sorted_active, x='Contacts', y='Messages')
        sb.barplot(data=df,y=sorted_active[-10:].index,x=sorted_active[-10:].values,palette="spring")
        j = -10 
        for i, v in enumerate(sorted_active.values[-10:]):
            plt.text(
                0, i + 0.2, str(sorted_active.index[j]), color="black", fontsize=10
            )
            j += 1'''
'''     else:
        fig = px.bar(sorted_active, x='Contacts', y='Messages')
        j = -1 * len(sorted_active.values)
        for i, v in enumerate(sorted_active.values):
            plt.text(
                0, i + 0.2, str(sorted_active.index[j]), color="black", fontsize=10
            )
            j += 1 ''' 
    
'''    plt.title("Most Active Memebers", fontsize=15)
    plt.yticks([], [])
    plt.xticks(fontsize=5)
    plt.ylabel("")
    sb.despine(left=True)
    plt.savefig(
        os.path.join("static/images/dashboard", filename + "mactive.png"),
        bbox_inches="tight",
    )'''

def monthsanal(filename):
    df = pd.read_csv(os.path.join("csvs", filename))
    plt.figure(figsize=(15, 10))
    df['message_count'] = [1] * df.shape[0] 
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    grouped_by_time = df.groupby('hour').sum().reset_index().sort_values(by = 'hour')
    
'''    sb.barplot(data=df,x=grouped_by_time.hour, y=grouped_by_time.message_count,palette="plasma")
    plt.title('Most Active Hours')
    sb.despine()
    plt.savefig(
        os.path.join("static/images/dashboard",filename + "mwise.png"),
        bbox_inches="tight",
    )'''

    
def get_emojis(text):
    emojis = []
    for msg in text:
        emojis.extend([c for c in msg if c in emoji.UNICODE_EMOJI['en']])

    emojis_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))),columns=['emoji','counts'])
    return emojis_df

def least(filename):
    df = pd.read_csv(os.path.join("csvs", filename))
    plt.figure(figsize=(15, 10))
    df['message_count'] = [1] * df.shape[0] 
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    grouped_by_time = df.groupby('hour').sum().reset_index().sort_values(by = 'hour')
    fig = px.bar(grouped_by_time,x=grouped_by_time.hour, y=grouped_by_time.message_count,title="Most Active Hours")
    fig.update_layout(xaxis_title='Hour', yaxis_title='Message Count')
    fig.write_html(os.path.join('static/',filename+'active.html'))
    monthg = Counter(pd.to_datetime(df["date_time"]).dt.month)
    od = collections.OrderedDict(sorted(monthg.items()))
    values = []
    for value in od.values():
        values.append(value)
    keys = []
    for key in od.keys():
        keys.append(key)
    months = [
        "",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    kib = []
    for k in keys:
        kib.append(months[k])
    fig = px.bar(df,x=kib, y=values,title="Most Active Month")
    fig.update_layout(xaxis_title='Month', yaxis_title='Message Count')
    fig.write_html(os.path.join('static/',filename+'mwise.html'))
    
    
    df['Cleaned Reviews'] = df['Messages'].apply(clean)
    df['POS tagged'] = df['Cleaned Reviews'].apply(token_stop_pos)
    df['Lemma'] = df['POS tagged'].apply(lemmatize)
    fin_data = pd.DataFrame(df[['Cleaned Reviews', 'Lemma']])
    fin_data['SWN analysis'] = df['POS tagged'].apply(sentiwordnetanalysis)
    swn_counts= fin_data['SWN analysis'].value_counts()
    
    fig = px.pie(values=swn_counts.values, names=swn_counts.index, title='Sentiment Scores')
    fig.write_html(os.path.join('static/',filename+'sentipie.html'))
    
    df1=df['Cleaned Reviews'].copy()
    df1
    str=''
    for i in df1:
        str+= i +"\n"
    text = str
    lower_case = text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))

# Using word_tokenize because it's faster than split()
    tokenized_words = word_tokenize(cleaned_text, "english")

# Removing Stop Words
    final_words = []
    for word in tokenized_words:
        if word not in stopwords.words('english'):
            final_words.append(word)

# Lemmatization - From plural to single + Base form of a word (example better-> good)
    lemma_words = []
    for word in final_words:
        word = WordNetLemmatizer().lemmatize(word)
        lemma_words.append(word)

    emotion_list = []
    with open('./emotions.txt','r',errors='ignore') as file:
        for line in file:
            clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
            word, emotion = clear_line.split(':')
            if word in lemma_words:
                emotion_list.append(emotion)

    w = Counter(emotion_list)
    
    
    fig = px.bar(df,x=w.keys(), y=w.values(),title="Emotion Analyis")
    fig.update_layout(xaxis_title='Month', yaxis_title='Message Count')
    fig.write_html(os.path.join('static/',filename+'sentibar.html'))


def week(filename):
    df = pd.read_csv(os.path.join("csvs", filename))
    plt.figure(figsize=(15, 10))
    weekday = Counter(pd.to_datetime(df["Date"]).dt.weekday)
    od = collections.OrderedDict(sorted(weekday.items()))
    values = []
    for value in od.values():
        values.append(value)
    keys = []
    for key in od.keys():
        keys.append(key)
    week = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    kib = []
    for k in keys:
        kib.append(week[k])

    sb.barplot(data=df,x=kib,y=values,palette="plasma")
    fig = px.bar(df,x=kib, y=values,title="WeekDay-wise Messages")
    fig.update_layout(xaxis_title='Month', yaxis_title='Message Count')
    fig.write_html(os.path.join('static/',filename+'week.html'))

    
    
    