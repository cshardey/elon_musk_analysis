import os
import tweepy
from dotenv import load_dotenv


def tweet_api():
    load_dotenv()
    consumer_key = os.getenv("API_KEY")
    consumer_secret = os.getenv("API_SECRET")
    access_token = os.getenv("ACCESS_TOKEN")
    access_token_secret = os.getenv("ACCESS_TOKEN_SECRET")
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    return api

