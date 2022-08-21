import re

import matplotlib
import nltk
import numpy as np
import pandas as pd
import plotly.io
# plotting
import seaborn as sns
from matplotlib import patches
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.sentiment import sentiment_analyzer, SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# nltk
from nltk.stem import WordNetLemmatizer
from gensim.parsing.preprocessing import STOPWORDS
# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

nltk.download('wordnet')
nltk.download('omw-1.4')
df = pd.read_csv('tt.csv')


# Remove stop words from dataframe text field
def remove_stop_words(df):
    # remove stop words
    text_list = []
    stop_words = set(stopwords.words('english'))
    stop_words.update(
        ['Twitter', 'RT', 'TwitterTakeover', 'Twitter', 'Elon', 'a', 'the', 'i', 'elon', 'musk', 'u', 'Twitter',
         'twitter', "dogitt '", 'twitterceo', 'come', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
         'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'before',
         'being', 'below', 'between', 'both', 'by', 'can', 'd', 'did', 'do',
         'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from',
         'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'twitterceo',
         'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
         'into', 'is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
         'me', 'more', 'most', 'my', 'myself', 'now', 'o', 'of', 'on', 'once',
         'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'own', 're', 's', 'same', 'she', "shes", 'should',
         "shouldve", 'so', 'some', 'such',
         't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
         'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
         'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was',
         'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'UFUF',
         'why', 'will', 'with', 'won', 'y', 'you', "youd", "youll", "youre",
         "youve", 'your', 'yours', 'yourself', 'yourselves'])
    text_list = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    # get only text column to a new dataframe with column text
    text_list = pd.DataFrame(df, columns=['text', 'created'])

    return text_list


# Remove punctuation from dataframe text field
def remove_punctuation(df):
    # remove punctuation
    df['text'] = df['text'].str.replace('[^\w\s]', '')
    return df


# Remove repeating characters from dataframe text field
def remove_repeating_characters(df):
    # remove repeating characters
    df['text'] = df['text'].str.replace('(.)\1+', '\g<1>')
    return df


# Remove urls from dataframe text field
def remove_urls(df):
    # remove urls
    df['text'] = df['text'].str.replace(r'http\S+', '')
    return df


def remove_definite_articles(df):
    # remove definite articles
    df['text'] = df['text'].str.replace('(?<!\w)(a|an|the)(?=\s\w)', '')
    return df


# Remove numbers from dataframe text field
def remove_numbers(df):
    # remove numbers
    df['text'] = df['text'].str.replace('\d+', '')
    return df


#  Tokenize dataframe text field using nltk
def tokenize_data(df):
    # tokenize dataframe text field
    df['text'] = df['text'].apply(lambda x: nltk.word_tokenize(x))
    return df


# Applying stemming to dataframe text field
def stemming_data(dataset):
    lemmatizer = WordNetLemmatizer()
    dataset['text'] = dataset['text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    dataset['text'].head()
    return dataset


def stemming_on_text(dataset):
    st = nltk.PorterStemmer()
    dataset['text'] = dataset['text'].apply(lambda x: [st.stem(word) for word in x])
    dataset['text'].head()
    return dataset


def create_sentiment_column(df):
    # create sentiment column and  fill it NEGATIVE, NEUTRAL, POSITIVE
    # create a new dataframe with the sentiment column
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
    # create a sentiment_type column and fill it NEGATIVE, NEUTRAL, POSITIVE
    df['Sentiment Type'] = df['sentiment'].apply(lambda x: 'NEGATIVE' if x < 0 else 'NEUTRAL' if x == 0 else 'POSITIVE')

    # convert create column to date
    df['created'] = pd.to_datetime(df['created'])
    # get the month of the created column
    df['month'] = df['created'].dt.month_name()

    # plot a graph of the sentiment_type column using sns with NEGATIVE blue, NEUTRAL yellow, POSITIVE red
    sns.set(font_scale=2)

    palette = {'NEGATIVE': '#F44336', 'NEUTRAL': '#039BE5', 'POSITIVE': '#388E3C'}
    sns.countplot(x='month', data=df, hue='Sentiment Type', palette=palette)
    # change xlabel to 'Month' and ylabel to 'Number of Tweets'
    plt.xlabel('Month')
    plt.ylabel('Number of Tweets')
    #increase font size
    plt.show()
    # make POSITIVE  green color and NEGATIVE red color and NEUTRAL blue color sns

    # plt.show()


    #
    #
    # sns.lineplot(data=df, x='created', y='tweet_volume', hue='sentiment_type')
    # plt.show()

    return df


def generate_wordcloud(df):
    # create a word cloud from da
    # pick positive words

    # SELECT ONLY POSITIVE WORDS
    positive_words = df[df['sentiment'] > 0.0]
    positive_words = positive_words['text']
    # select text column of the dataframe

    print(positive_words.head(20))
    plt.figure(figsize=(20, 20))
    wc = WordCloud(background_color='white', max_words=2000, max_font_size=50, scale=3,
                   random_state=42).generate(' '.join(str(v) for v in positive_words))
    plt.imshow(wc)
    plt.axis('off')
    plt.show()


def print_word_cloud(dataset):
    # create a word cloud from da
    # pick positive words

    # SELECT ONLY POSITIVE WORDS
    positive_words = dataset[dataset['sentiment'] > 0.0]

    text = ', '.join('%s' % v for v in positive_words)
    stop_words = set(stopwords.words('english'))

    wordcloud = WordCloud(width=1600, height=800, collocations=False,
                          background_color='white', stopwords=STOPWORDS).generate(text)
    # Configure for a bigger image size
    plt.figure()
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()


def print_word_count(dataset):
    # get positive words
    positive_words = dataset[dataset['sentiment'] > 0.0]
    positive_words = positive_words['text']
    # for each word in the positive words, count the number of times it appears
    word_count = {}
    stp_words = ['twittertakeov', 'twitter', 'elonmusk', 'elon', 'ufcc', 'nft', 'musk', 'the', 'uf', 'i', 'x', 'get',
                 'guy', 'thi', 'follow', 'want', 'what',
                 'chang', 'it', 'see', 'elonmuskbuytwitt']
    for word in positive_words:
        for word_ in word:
            # remove from word_count if it is a stop word
            if word_ not in stp_words and len(word_) > 3:
                if word_ not in word_count:
                    word_count[word_] = 1
                else:
                    word_count[word_] += 1

    # sort the word count by the number of times it appears
    word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    # join the count of free and speech words to freespeech
    # add freespeech  word to the word_count l

    # get top 20 words to list
    top_20_words = word_count[:1000]
    # search the top_20 list for the word 'freespeech'
    # if it is found, replace the word with 'freespeech' and the count with the sum of the count of free and speech
    for i in range(len(top_20_words)):
        if top_20_words[i][0] == 'free' or top_20_words[i][0] == 'speech':
            top_20_words[i] = ('freespeech', top_20_words[i][1] + top_20_words[i + 1][1])
            top_20_words.pop(i + 1)
            break

    # rearrange list  by highest count
    top_1000 = sorted(top_20_words, key=lambda x: x[1], reverse=True)[:1000]

    print(top_20_words)

    top_words = sorted(top_20_words, key=lambda x: x[1], reverse=True)

    # print wordcloud
    plt.figure(figsize=(20, 20))
    wc = WordCloud(background_color='white', max_words=1000, max_font_size=50, scale=3,
                   random_state=42).generate(' '.join(str(v[0]) for v in top_words))
    plt.imshow(wc)
    plt.axis('off')
    plt.show()

    # arrange in
    # plot horizontal bar chart  using seaborn with different colors for  words
    sns.set(style='whitegrid')
    plt.figure(figsize=(20, 20))
    sns.barplot(y=[i[0] for i in top_20_words], x=[i[1] for i in top_20_words], palette='husl')
    # show legend on bar chart
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
    plt.show()

    # top


new_df = remove_stop_words(df)
print('Stop words removed', new_df.head(10))

def_art = remove_definite_articles(new_df)
print('Definite articles removed', def_art.head(10))

pun_df = remove_punctuation(def_art)

print('Punctuation removed', pun_df.head(10))

rpt_df = remove_repeating_characters(pun_df)
print('Repeating characters removed', rpt_df.head(10))

num_df = remove_numbers(rpt_df)
print('Numbers removed', num_df.head(10))

url_df = remove_urls(num_df)

print('Urls removed', url_df.head(10))

senTdf = create_sentiment_column(url_df)
print('Sentiment column created', senTdf.head(10))

token_df = tokenize_data(senTdf)
print('Tokenized', token_df.head(10))

steam_df = stemming_data(token_df)
print('Stemmed', steam_df.head(10))

stem_df = stemming_on_text(steam_df)

print('Stemmed on text', stem_df.head(10))

print_word_count(stem_df)

generate_wordcloud(stem_df)
