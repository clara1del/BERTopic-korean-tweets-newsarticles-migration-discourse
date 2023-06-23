#code for visual studio code, python
#BERTopic model, in one step
#before starting, conda activate environment

# %% 

import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from konlpy.tag import Mecab
from bertopic import BERTopic
import matplotlib.pylab as plt
from bertopic.representation import MaximalMarginalRelevance
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer
import glob
import os
from datetime import datetime
import dateutil.parser as parser  

df=pd.read_csv('tweets2012.csv', parse_dates=['Date'], lineterminator='\n')  #replace with own dataset

df=df.dropna()

timestamps = df['Date'].tolist()

#function to clean text from unwanted characters
file = open('stopwords.txt', 'r')  # list of Korean stopwords is in this github project
words = file.read()
file.close()
stopwords = list(words.split())

def remove_stopwords(text):
    text = [word for word in text.lower().split() if word not in stopwords]
    text = ' '.join(text[0:])
    return text

def clean_text(text):
    #Remove hyper links
    text = re.sub(r'https?:\/\/\S+', ' ', text)
    #Remove @mentions
    text = re.sub(r'@[A-Za-z0-9]+', ' ', text) #(if the below works, we can erase this)
    # Remove user @ references and '#' from tweet
    text = re.sub(r'\@\w+|\#|\d+', ' ', text)
    # Remove noise
    text = re.sub(r'[-_:,\'"+RT]|[a-z]|[📍📌💬🌏]|[✔▶]|\[|\]|\*|\(|\)|\.\.\.+|[…]|\.\.+', ' ', text)
    # Remove extra brakets
    text=text.strip()
    # Remove urls
    text = re.sub(r"http\S+|www\S+|https\S+", ' ', text, flags=re.MULTILINE)
    # Remove Stop words
    text=remove_stopwords(text)
    return text

# Apply the clean_text function to the 'Tweets' column
df['Tweets']=df['Raw_Tweet'].apply(clean_text)

df.head()

documents = [line.strip() for line in df.Tweets]
preprocessed_documents = []

for line in tqdm(documents):
  if line and not line.replace(' ', '').isdecimal():
    preprocessed_documents.append(line)

text=''.join(preprocessed_documents[:10000])
text=text.split('.')

#We apply BERTopic to Korean text, and thus want to use Mecab as a tokenizer instead
class CustomTokenizer:
    def __init__(self, tagger):
        self.tagger = tagger
    def __call__(self, sent):
        sent = sent[:1000000]
        word_tokens = self.tagger.nouns(sent) #we keep only the nouns for clarity with tagger.nouns . To keep all words: word_tokens = self.tagger.morphs(sent)
        result = [word for word in word_tokens if len(word) > 1]
        return result

#This seed topic list pushes the model towards the defined topics, IF they exist. 

seed_topic_list = [['실업', '일자리', '작업 경쟁','취업 경쟁','직장 잃','경쟁'], #1. Common Arguments against Immigration: “immigrants take jobs, lower wages, hurt the poor” (ref: https://www.cato.org/blog/14-most-common-arguments-against-immigration-why-theyre-wrong)
                   ['건강 보험', '병원', '복지','보건의료사비스', '보건의료', '보건의료요구','의료 혜택',  '보험 혜택',  '의료', '건강', '의료혜택','보건의료정책','IHS', 'NHS','의료 서비스', '사회복지'], #2. Common Arguments against Immigration : “ abuse welfare”
                   ['세금','불경기'], #3. Common Arguments against Immigration: “increase budget deficit and government debt”
                   ['한국어 실력', '문화 교류','통합','언어장벽', '동화','사회통합 프로그램','공동'], #4.Common Arguments against Immigration: “ don’t assimilate, integrate”community
                   ['불법','범죄','위험', '살인', '강도', '절도', '매춘', '마약', '사이버 범죄', '텔레뱅킹 사기', '피싱', '외국인 범죄', '위조범', '밀수품', '산업연수생 범죄'], #5. Common Arguments against Immigration: “source of crime”
                   ['테러','테러분자','테러리즘','테러리스트','사보타주'], #6.Common Arguments against Immigration: “terrorism” 
                   ['민족주의','한국적 가치관, 한국성','국가 이미지','한국 이미지'], #7.Common Arguments against Immigration: national sovereignty 
                   ['정부','법무부','윤석열','문재인','박근혜','이명박','노무현','김대중','대통령','통치'], #8.ruling class - government
                   ['국적','교포', '미국', '일본', '이집트',  '주전자','고려인', '러시아','흑인','아랍','라틴아메리카 ' , '베트남', '둥포', '백인', '조선족', '러시아인', '미국인', '유럽인', '서구인','서양인', '동남아시아인', '동남아인', '우즈벡인', '우즈베키스탄 이주', '중국인', '중국', '아프리카', '인도', '우크라이나',  '중동', '몽골인', '몽골', '탈북이주민', '북한이탈주민'], #9.dividing the working class with identity politics:racism and ethnicity 인종
                   ['이주여성','여성이주노동자','여성','여자', '젠더'], #10.dividing the working class with identity politics: gender discrimination 성별
                   ['MTU', '조합', '이주노동희망센터', '이민자 센터', '연대', '이주노동자노동조합', '이주노조', '이주노동자 노동조합', '이주민센터 친구','상담'], #11.uniting the working class with union
                   ['국제결혼', '외국인 신부', '결혼이민자', '결혼이민','결혼이주자', '결혼 중개업', '이민자 부모','결혼', '이혼', '아내', '남편', '신부', '가정 폭력', '가족 폭력'], #12.dividing migrants by migrant status and place in the online migration debate: marriage
                   ['가족','다문화주의','외국인 아동','다문화 가정','이민자 부모', '임산부','임신','어린이', '부모님','다문화가족'], #13.Family
                   ['무슬림','이슬람교도','이슬람','무슬리마'], #14. divide by religion
                   ['선생님','영어 선생님', '부자', '사업가','투자자','교수','상용 비자'], #15.status: high payed worker
                   ['농장','건설','선박', '어업','E9', '고용허가제', '서비스업', '농축산업', '건설업','제조업','건설공사', '작물재배업','축산업','양식어업','소금채취업', ' 비전문취업','건설폐기물 처리업','육체노동','공장', '건설노동자', '계절 노동자', '3D 업종','산업연수생 시스템'], #16.status: low payed workers in 3d industries
                   ['가사도우미', '입주도우미', '육아도우미' , '간병도우미', '베이비시터', '외국인 가사도우미', '돌봄도우미', '간병인', '도우미','요식업','식업'], #17.status: gendered work in service industry, nurse, cleaning staff
                   ['관광객', '여행', '문화', '여행자','관광'], #18.status: tourist
                   ['학생', '학교', '대학교', '대학생', '교환 학생','유학비자', '유학', '어학연수', '교환학생', '연구유학'], #19.status: student
                   ['탈북자','북한이탈주민','탈북','탈북자','탈북민','새터민','북한이탈주민'], #20.north korea refugees
                   ['불법체류자','불법체류 외국인', '불법체류', '미등록', '미등록 이주자', '불쳬자','미등록외국인근로자', '외국인 불법 근로자','무허가 노동자'], #21.status: undocumented immigration
                   ['예멘', '미얀마','파키스탄','방글라데시','에티오피아'], #22.status: refugee based on nationality
                   ['변호사', '법원','비자', '법', '이민법','시민권', '노동법','국적법', '이민 정책', '근로기준법'], #23.immigration law
                   ['출입국관리소', '비자연장', '비자유형변경', '비자신청', '영주권', '체류허가', '비자'], #24.administration
                   ['저임금 노동', '값싼 노동자', '저임금', '최저임금','고용','계급','자본'], #25.wages and work, working class in capitalism
                   ['노동 착취','남용', '착취', '사고', '직장 괴롭힘', '괴롭힘', '근로환경', '작업 조건','폭력','노예'], #26.human rights abuse
                   ['노동시간', '복지', '시설', '산업재해', '임금체불', '시간외 수당', '해고','부상 보상'], #27.working conditions
                   ['경찰단속', '단속','합동단속', '정부합동단속', '단속추방','추방','강제추방','외국인보호소','경찰', '감옥','국경검사','한국경찰','구속'], #28.police state, border, deportation, expulsion
                   ['차별','차별금지','외국인 혐오','소수자',' 배제','불평등','계층','고정관념','낙인','선입견','인종차별','기본권', '평등', '불평등', '편견','인권'], #29.discrimination awareness 
                   ['경제', '경제이주', '경제적 이득', '노동수요','노동력 부족','이윤'], #30.economy    
                   ['비자','신청','발급'], #visa administration (added topic)
                   ['관광','여행'], #(added topic)
                   ['뉴스'], #(added topic)
                   ['망명 신청자','난민','피난자'], #refugees
                   ['친구'], #(added topic)
                   ['자발'], #(added topic)
                   ['종목'], #(added topic)
                   ['영어','한국어'], #(added topic)
                   ['외국인','외국'], #(added topic)
                   ['한국','한국인'], #(added topic)
                   ['일본','일본어'], #(added topic)
                   ['이주노','동자'], #(added topic)
                   ['개인','사람','나라','카드','생각']] #(added topic to decrease the overlapping of topics)

    
custom_tokenizer = CustomTokenizer(Mecab())
vectorizer = CountVectorizer(tokenizer=custom_tokenizer, max_features=3000)

representation_model = MaximalMarginalRelevance(diversity=0.3)

umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine') #reduce dimensionality # neighbords, components, can be updated to decrease topic overlapping

model = BERTopic(embedding_model="sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens", \  #we use this model for Korean text
                 vectorizer_model=vectorizer,
                 umap_model=umap_model,      # Reduce dimensionality
                 seed_topic_list=seed_topic_list,
                 min_topic_size=35, 
                 nr_topics=31, #can be changed to "auto"
                 top_n_words=20, 
                 calculate_probabilities=True)


topics, probs = model.fit_transform(documents) # Train the BERTopic model
new_topics = model.reduce_outliers(documents, topics) # Reduce outliers

model.save('topics2012.h5')

# %% 

#Now for the results

for i in range(0, 32): 
  print(i,'번째 토픽 :', model.get_topic(i))

# %% 
#To find the most similar topics for a specific word (here, "여성" )

similar_topics, similarity = \
model.find_topics("여성", top_n = 10) 

print("Most Similar Topic Info: \n{}".format(model.get_topic(similar_topics[0])))
print("Similarity Score: {}".format(similarity[0]))

print("\n Most Similar Topic Info: \n{}".format(model.get_topic(similar_topics[1])))
print("Similarity Score: {}".format(similarity[1]))

print("\n Most Similar Topic Info: \n{}".format(model.get_topic(similar_topics[2])))
print("Similarity Score: {}".format(similarity[2]))


# %%
model.get_representative_docs()
model.get_topic_info()
model.get_topics()
model.generate_topic_labels()

# %%
#Visualization of the results: how to plot an interactive graph html

import chart_studio.plotly as py
import plotly.graph_objects as go

fig = model.visualize_barchart()
fig.write_html("2012barchart.html")
fig.show()

fig = model.visualize_topics()
fig.write_html("2012topics.html")
fig.show()

fig = model.visualize_hierarchy()
fig.write_html("2012hierarchy.html")
fig.show()

fig = model.visualize_heatmap()
fig.write_html("2012heatmap.html")
fig.show()

fig = model.visualize_term_rank()
fig.write_html("2012termrank.html")
fig.show()

fig = model.visualize_term_rank(log_scale=True)
fig.write_html("2012logtermrank.html")
fig.show()


# %%
#the visualization of the topics overtime takes longer, so it is separated into another cell

import chart_studio.plotly as py
import plotly.graph_objects as go

topics_over_time = model.topics_over_time(documents, timestamps)
fig = model.visualize_topics_over_time(topics_over_time)
fig.write_html("2012topicstime.html")
fig.show()


# %%
#If we only want images, no interactvie graphs html, visualization can be done as:

topics_over_time = model.topics_over_time(documents, timestamps)
model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
model.visualize_topics_over_time(topics_over_time)


model.visualize_topics()
model.visualize_distribution(probs[0])
model.get_representative_docs()
model.get_topic_info()
model.get_topics()
model.visualize_barchart()
model.visualize_hierarchy()
model.visualize_heatmap()
model.visualize_term_rank()
model.visualize_term_rank(log_scale=True)
model.generate_topic_labels()
