# coding: utf-8
#scraping news articles with webdriver and selenium
#code for jupyter notebook (in order to use selenium and webchromedriver)

#!pip install selenium
#!install chromium chromium-driver


from bs4 import BeautifulSoup
import pandas as pd
import time
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

driver.get("https://www.chosun.com/nsearch/?query=%EC%9D%B4%EC%A3%BC%EB%85%B8%EB%8F%99%EC%9E%90") #replace with link of your query

titles = []
links = []
dates = []
until_pages = 10000
for ipage in range(until_pages+1)[1:]:   #replace with your vocabulary query (here 이주노동자) 
    driver.get("https://www.chosun.com/nsearch/?query=이주노동자)&page="+str(ipage)+"&siteid=&sort=1&date_period=all&date_start=&date_end=&writer=&field=&emd_word=&expt_word=&opt_chk=false&app_check=0&website=www,chosun&category=")
    content = driver.page_source.encode('utf-8').strip()
    soup = BeautifulSoup(content)
    b = soup.find_all("div", "story-card-right | grid__col--sm-8 grid__col--md-8 grid__col--lg-8 box--pad-left-xs")

    counter = 1
    for i in b:

        titles.append(i.find("a", "text__link story-card__headline | box--margin-none text--black font--primary h3 text--left").find("span").get_text())
        links.append(i.find("a", "text__link story-card__headline | box--margin-none text--black font--primary h3 text--left").get("href"))
        c = i.find("div","story-card__breadcrumb | text--grey-60 font--primary font--size-sm-14 font--size-md-14 box--margin-bottom-sm text--line-height-1.43")
        d = c.find_all("span")
        date = d[len(d)-1].get_text()
        dates.append(date)
kamus = {"date" : dates, "title": titles, "link" : links}
df = pd.DataFrame(kamus)
df


#next cell


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


# next cell

df["article text"] = article_texts
df.to_excel("Downloads/chosun_workmigrants.xlsx")
df.to_csv("Downloads/chosun_workmigrants.csv")





