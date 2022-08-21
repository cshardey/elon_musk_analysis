import re

import emoji
import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import STOPWORDS
from wordcloud import WordCloud

from tweet_auth import tweet_api
import tweepy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from bs4 import BeautifulSoup


def tweet_data():
    api = tweet_api()
    start_date = '202201180000'
    end_date = '202204190000'
    # Get tweets with hashtag #TwitterTakeover and ##TwitterSoldusing tdate_range
    tweets = tweepy.Cursor(api.search_full_archive, query='#TwitterCEO',
                           fromDate=start_date, label='Test').items(1000)

    # print length of iterator



    # create a dataframe with the tweets
    df = pd.DataFrame(columns=['tweetId', 'text' ])
    # loop through the tweets and add them to the dataframe
    for tweet in tweets:
        df = df.append(
            {
                'id': tweet.id,
                'text': tweet.text,
                'created_at': tweet.created_at,
                'user_location': tweet.user.location,
                'user_listed_count': tweet.user.listed_count,
                'user_favourites_count': tweet.user.favourites_count,
                'user_geo_enabled': tweet.user.geo_enabled,
                'user_lang': tweet.user.lang,

            },
            ignore_index=True,
        )

    print("Length of ", len(df))
    df.to_csv('tweets_sold.csv', index=False)
    return df




def token_sentiment_analysis(df):
    # remove stopwords from tweet text
    # create sentiment column and  fill it NEGATIVE, NEUTRAL, POSITIVE
    # create a new dataframe with the sentiment column
    sia = SentimentIntensityAnalyzer()

    
    text = df.text.tolist()


  # join the list and lowercase all the words
    text = ' '.join(text).lower()

    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white').generate(text)

    # Configure for a bigger image size
    plt.figure()
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()

    return df


def clean_tweet_data(dataframe):


    # Remove all html characters using beautifulsoup
    dataframe['text'] = dataframe['text'].apply(lambda x: BeautifulSoup(x, "lxml").get_text())




    return dataframe







token_sentiment_analysis(clean_tweet_data(dataframe=pd.read_csv('tc.csv')))