The shaping of the narrative on migration:
Using BERTopic and Mecab for a discourse analysis of the framing of the concept of migration from a Korean corpus

The codes and corpora in this repository are from a project exploring the shaping of public opinion on migration in South Korea by utilizing BERT topic modelings (Dimo 2020; Grootendorst 2022). Data are the public discourse on Twitter and the three biggest local newspapers. 

The findings through BERTopic modeling as a tool of discourse analysis on large data shows that, rather than a simple overall negative media narrative, the news outlets create distinctive concepts of migrants, fragmented into clustered groups, alienated from each other based on their social identities, migration status, and citizenship status. Discriminatory tropes (such as a criminalization frame and a victimization frame) predominant in the Mass Media corpus, are less salient in the New Media corpus and the Public Opinion (Tweets) corpus, where topics of compassion, human rights, union, reports of shared experiences, desire to share culture and communicate, are predominant.

With the c-TF IDF formula giving the significance of words per topic,  the creation of a divisive concept of refugees is visualized, with the fragmentation of one group (for example, refugees) into vastly distanced topics (either in the victimization frame, with "kid" and "refugee" in one cluster, or the criminalization frame, with "refugee" and "terrorism" in one cluster).
This division in the public narrative supports the division in the governemental policies. For example, the Ministry of Justice divides asylum seekers applying for a refugee Visa into "humanitarian" or "economic" refugee categories. Asylum seekers placed in the "economic" refugee category are denied the refugee status.

The intertopic distance maps illustrates this shaping of divisive semantic meanings.


.BERTopic for Korean Data

The Topic model BERTopic is performed with a Mecab package from konlpy.

For more information on BERTopic, please check: https://github.com/MaartenGr/BERTopic

For more information on Mecab (for Google colab), please check: https://github.com/SOMJANG/Mecab-ko-for-Google-Colab

For more information on konlpy, please check: https://github.com/konlpy/konlpy/

.Corpus

The corpus of tweets has been scraped with the scraper snscrape.

For more information on snscrape, please check: https://github.com/JustAnotherArchivist/snscrape

The corpus of news articles has been scraped with webchromedriver.

The corpus of descriptions of news articles has been scraped with the Naver API.

mecab.nouns function is used.


![workflow](https://github.com/clara1del/BERTopic-korean-tweets-newsarticles-migration-discourse/assets/120312491/19583f71-40c1-4ff0-a2f1-5aea0795fe4d)


.Results

Results are limited by the overlapping of topics. Some examples of the visualization are:

![1new22_23map](https://github.com/clara1del/BERTopic-korean-tweets-newsarticles-migration-discourse/assets/120312491/9b8babd5-2d47-4d27-b9ac-11bcb20e0efb)

![massmedia2009_23_inter](https://github.com/clara1del/BERTopic-korean-tweets-newsarticles-migration-discourse/assets/120312491/a2067ed1-5bfc-496a-a4aa-4a0fcc7373c8)



