# -*- coding: utf-8 -*-
#code for google colab
##### Scraper with Official API
#Scraping descriptions of news articles published in Naver News, using the official Naver API. 
#Only the 1000 most recent articles can be scraped
#Only a short description, news title, publication date, can be scraped
#Articles are scraped according to word queries

import pandas as pd
import os
import sys
import urllib.request

client_id = "YOUR CLIENT ID"  #first, we must register for the (free) Naver News API, and obtain an ID and password
client_secret = "YOUR PASSWORD"

file_path = "./tesst.json"
encText = urllib.parse.quote("이주노동자") #here write the vocabulary query (here, "이주노동자" )
url = "https://openapi.naver.com/v1/search/news.json?display=100&start=1000&sort=sim&query=" + encText    #100 display and 1000 start can be modified according to the time window

request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)
response = urllib.request.urlopen(request)
rescode = response.getcode()
if(rescode==200):
    response_body = response.read()
    print(response_body.decode('utf-8'))
else:
    print("Error Code:" + rescode)

import pandas as pd
df = pd.read_fwf('/content/drive/MyDrive/이주노동자.txt')
df.to_csv('/content/drive/MyDrive/immigrantworkersnewstitles.csv')

##### Organize the data

from google.colab import drive
drive.mount('/content/drive')
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/immigrantworkersnewstitles.csv', lineterminator='\n')

column_names = list(df.columns.values)

for column_headers in df.columns:
    print(column_headers)

df = df.drop(df.columns[[0, 1, 2]], axis=1)
df.rename(columns={"Unnamed: 2": "newstitle"}, inplace=True)

df2 = df[df['newstitle'].str.contains('description|pubDate', na = False)]
df2.to_csv('/content/drive/MyDrive/datedimmigrantworkersnewstitles.csv')



