# -*- coding: utf-8 -*-
#scraping tweets using snscrape
#unlike the official Twitter API, there are no restrictions on the number of tweets
#however, no information on the user (such as geolocalization) is available
#tweets are scraped according to keyword queries

#in command: pip3 install snscrape

#snscrape --jsonl --progress --max-results 100000 twitter-search "xxxx since:2009-12-01 until:2022-07-26" > xxxxtweets.json 

#for example:
#snscrape --jsonl --progress --max-results 1000000 twitter-search "이주민 since:2009-12-01 until:2022-07-26" > ijoumintweets.json

#saves the data as a json file


#### Organize the data

#code for google colab
#add json file to to google drive
#add the json file to the collab environment


import pandas as pd
tweets_df = pd.read_json('ijoumintweets.json', lines=True)

attributes_container = []

for i in range(tweets_df.shape[0]):
    if i==tweets_df.shape[0]:
      break
    attributes_container.append([tweets_df.loc[i]['date'], tweets_df.loc[i]['likeCount'], tweets_df.loc[i]['content']])

# Creating a dataframe to load the list
tweets_df = pd.DataFrame(attributes_container, columns=["Date", "Likes", "Tweet"])
tweets_df

