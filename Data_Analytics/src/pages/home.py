import pathlib
import utils.display as udisp
import pandas as pd
import calendar
import streamlit as st
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
	
import csv
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from collections import Counter
from wordcloud import WordCloud
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
import os

from collections import Counter
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def write():
    udisp.title_awesome("Barbenheimer Data Analytics")
    
    
    keys = {
    "Barbenheimer Review Dataset": './data/barbenheimer.csv'
            }
    image = Image.open('./img/1.jpg')
    st.image(image, use_column_width=True)
    st.header("Dataset")
    pick = st.selectbox("Select Dataset: ", list(keys))
    
        
    df = pd.read_csv(keys[pick], encoding='utf8')
    
    # Enriching Data
    df['date'] = pd.to_datetime(df['date'])
    df['Year']=df['date'].dt.year 
    df['Day']=df['date'].dt.dayofweek         
    analyser = SentimentIntensityAnalyzer()
    df['Sentiment_Analysis'] = df['review'].apply(analyser.polarity_scores)
    df['neg'] = [analyser.polarity_scores(x)['neg'] for x in df['review']]
    df['neu'] = [analyser.polarity_scores(x)['neu'] for x in df['review']]
    df['pos'] = [analyser.polarity_scores(x)['pos'] for x in df['review']]
    df['Sentiment'] = np.where(df['neg'] >= df['pos'], 'Negative', 'Positive') 	
    st.dataframe(df)
    
    
    
    
    st.subheader("Total Number of Reviews")
    st.info(str(df.shape[0]))
    st.header("")
    st.header("")
    st.header("")
    st.header("")
    
    
    lists={
    'label',
    'score',
    'Day',
    'Sentiment'
    }
    st.header("Choose a General Attribute to Visualize")
    pick_lst = st.selectbox("Choose: ", list(lists))
    
    df3 = df.groupby(pick_lst).count()
    df3 = df3[['title']]
    df3.columns = ['Count']
    df3['x-axis'] = df3.index
    
    
    
    st.header("")
    st.subheader("Pie Chart of " + str(pick_lst))
    
    fig1 = go.Figure(data=[go.Pie(labels=df3['x-axis'], values=df3['Count'])])
    st.plotly_chart(fig1)
    

        
        
        
    lists2={
    'Positive',
    'Negative',
    }
    st.header("Choose a Sentiment to Summarize")
    pick_lst2 = st.selectbox("Choose: ", list(lists2))
    
 
    lang = 'english'
    count = 10
    
    if pick_lst2 == 'Positive':
    	positive_reviews = df.loc[df['Sentiment'] == 'Positive'] 
    	pos_rev = positive_reviews['review'].tolist()
    	corpus = ' '.join(pos_rev)
    	new_string = corpus.replace('.', '. ').strip()
    	lsa = LsaSummarizer(Stemmer(lang))
    	lsa.stop_words = get_stop_words(lang)
    	parser = PlaintextParser.from_string(new_string, Tokenizer(lang))
    	lsa_summary = lsa(parser.document, count)
    	lsa_s = [str(sent) for sent in lsa_summary]
    	summary = ' '.join(lsa_s)
    	st.header("Summary of Positive Reviews")
    	st.write(summary)
        
        
    if pick_lst2 == 'Negative':
        negative_reviews = df.loc[df['Sentiment'] == 'Negative'] 
        neg_rev = negative_reviews['review'].tolist()
        corpus = ' '.join(neg_rev)
        new_string = corpus.replace('.', '. ').strip()
        lsa = LsaSummarizer(Stemmer(lang))
        lsa.stop_words = get_stop_words(lang)
        parser = PlaintextParser.from_string(new_string, Tokenizer(lang))
        lsa_summary = lsa(parser.document, count)
        lsa_s = [str(sent) for sent in lsa_summary]
        summary = ' '.join(lsa_s)
        st.header("Summary of Negative Reviews")
        st.write(summary)
     

       
        
        
        
        

    
    #st.header("")
    #st.subheader("Pie Chart of " + str(pick_grp) + " by count" )
    
    #fig1 = go.Figure(data=[go.Pie(labels=df3['x-axis'], values=df3['Count'])])
    #st.plotly_chart(fig1)
