#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install selenium
#!install chromium chromium-driver


# In[2]:


#'https://www.chosun.com/nsearch/?query=%EB%82%9C%EB%AF%BC/', #(난민) = refugees = 2023-04-05 = 12799건 articles
#'https://www.chosun.com/nsearch/?query=%EC%9D%B4%EC%A3%BC%EB%AF%BC/'), #(이주민) = migrants = 2023-04-05 = 2448건 articles
#'https://www.chosun.com/nsearch/?query=%EC%9D%B4%EC%A3%BC%EC%97%AC%EC%84%B1/'), #(이주여성) = migrant women = 2023-04-05 = 1218건 articles
#'https://www.chosun.com/nsearch/?query=%EC%9D%B4%EB%AF%BC%EC%9E%90'), #(이민자) = migrants = 2023-04-05 = 8017건 articles
#'https://www.chosun.com/nsearch/?query=%EC%9D%B4%EC%A3%BC%EB%85%B8%EB%8F%99%EC%9E%90'), #(이주노동자) = migrant workers = 2023-04-05 = 1317건 articles
#'https://www.chosun.com/nsearch/?query=%EB%B6%88%EB%B2%95%EC%B2%B4%EB%A5%98%EC%9E%90'), #(불법체류자) = irregular migrants  = 2023-04-05 = 2158건 articles
#'https://www.chosun.com/nsearch/?query=%EC%99%B8%EA%B5%AD%EC%9D%B8%EB%85%B8%EB%8F%99%EC%9E%90'),#(외국인노동자) = foreign workers = 2023-04-05 = 2781건 articles
#'https://www.chosun.com/nsearch/?query=%EA%B2%B0%ED%98%BC%EC%9D%B4%EB%AF%BC%EC%9E%90'), #(결혼이민자) = marriage migrants = 2023-04-05 = 584건 articles
#'https://www.chosun.com/nsearch/?query=%EC%9D%B4%EC%A3%BC%EB%85%B8%EB%8F%99%EC%9E%90'), #(이주노동자) = migrant workers = 2023-04-05 = 1317 articles
#'https://www.chosun.com/nsearch/?query=%EB%AF%B8%EB%93%B1%EB%A1%9D%EC%9D%B4%EC%A3%BC%EB%AF%BC'), #(미등록이주민) = irregular migrants = 2023-04-05 = 10 articles
#'https://www.chosun.com/nsearch/?query=%EA%B2%B0%ED%98%BC%EC%9D%B4%EB%AF%BC'), #(결혼이민) = marriage migrants = 2023-04-05 = 386 articles
#'https://www.chosun.com/nsearch/?query=%EA%B2%B0%ED%98%BC%EC%9D%B4%EC%A3%BC%EC%9E%90'), #(결혼이주자) = marriage migrants = 2023-04-05 = 42 articles
#'https://www.chosun.com/nsearch/?query=%EC%99%B8%EA%B5%AD%EC%9D%B8' #(외국인) = foreigners = 2023-04-05 = 254852건 articles


# In[3]:


from bs4 import BeautifulSoup
import pandas as pd
import time


# In[4]:


from selenium import webdriver
def web_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--verbose")
    options.add_argument('--no-sandbox')
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument("--window-size=1920, 1200")
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    return driver

driver = web_driver()

#driver.get("https://www.chosun.com/nsearch/?query=%EB%82%9C%EB%AF%BC")
driver.get("https://www.chosun.com/nsearch/?query=%EC%9D%B4%EC%A3%BC%EB%85%B8%EB%8F%99%EC%9E%90")


# In[5]:


titles = []
links = []
dates = []
until_pages = 10000
for ipage in range(until_pages+1)[1:]:
    driver.get("https://www.chosun.com/nsearch/?query=이주노동자)&page="+str(ipage)+"&siteid=&sort=1&date_period=all&date_start=&date_end=&writer=&field=&emd_word=&expt_word=&opt_chk=false&app_check=0&website=www,chosun&category=")
    content = driver.page_source.encode('utf-8').strip()
    soup = BeautifulSoup(content)
    b = soup.find_all("div", "story-card-right | grid__col--sm-8 grid__col--md-8 grid__col--lg-8 box--pad-left-xs")
    # b = soup.find_all("a", "text__link story-card__headline | box--margin-none text--black font--primary h3 text--left")
    counter = 1
    for i in b:
        # counter =counter + 1
        # if counter%2 == 0:
        #     continue
        titles.append(i.find("a", "text__link story-card__headline | box--margin-none text--black font--primary h3 text--left").find("span").get_text())
        links.append(i.find("a", "text__link story-card__headline | box--margin-none text--black font--primary h3 text--left").get("href"))
        c = i.find("div","story-card__breadcrumb | text--grey-60 font--primary font--size-sm-14 font--size-md-14 box--margin-bottom-sm text--line-height-1.43")
        d = c.find_all("span")
        date = d[len(d)-1].get_text()
        dates.append(date)
kamus = {"date" : dates, "title": titles, "link" : links}
df = pd.DataFrame(kamus)
df


# In[6]:


iter = 0
article_texts = []
for ilink in df['link']:
    iter = iter + 1
    print(iter)
    # print(ilink)
    driver.get(ilink)
    try:
        content = driver.page_source.encode('utf-8').strip()
        soup = BeautifulSoup(content)
        b = soup.find("section","article-body")
        c = b.find_all("p")
        article_text = ""
        for ic in c:
            article_text = article_text + ic.get_text() + "\n\n"
        article_texts.append(article_text[:-2])
    except:
      try:
        b = soup.find("div","article-body")
        c = b.find_all("p")
        article_text = ""
        for ic in c:
            article_text = article_text + ic.get_text() + "\n\n"
        article_texts.append(article_text[:-2])
      except:
        article_texts.append("")


# In[7]:


df["article text"] = article_texts


# In[8]:


df.to_excel("/Users/X423/Downloads/chosun_workmigrants.xlsx")
df


# In[9]:


df.to_csv('/Users/X423/Downloads/chosun_workmigrants.csv')


# In[ ]:




