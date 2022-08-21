import twint
import nest_asyncio

def collect_tweets_from_user(tw_username, keywords, from_date, dst_file, limit = 50):

  # To avoid "the event loop is already running in python" error message.
  nest_asyncio.apply()

  # Configure
  conf = twint.Config()

  # Tweets from Joe Biden
  conf.Username = tw_username

  #Get tweets published after March 1st, 2022
  conf.Since = from_date

  # Search Criteria
  conf.Search = keywords
  conf.Limit = limit

  # Tweets Storage
  conf.Output = dst_file
  conf.Store_json = True

  # Run the execution
  twint.run.Search(conf)


tw_username = "JoeBiden"
keywords = "gas prices"
from_date = "2020-03-1 20:30:15"
dst_file = "./Biden_Gas_Prices.json"

collect_tweets_from_user(tw_username, keywords, from_date, dst_file)