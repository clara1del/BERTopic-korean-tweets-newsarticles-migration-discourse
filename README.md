The shaping of the narrative on migration: A corpus assisted quantitative discourse analysis of the impact of the divisive media framing of migrants in Korea

This work explores the shaping of public opinion on migration in South Korea by utilizing BERT topic modelings (Dimo 2020; Grootendorst 2022). Data are the public discourse on Twitter and the three biggest local newspapers. The study examines the content of these topics, highlighting key themes and their implications. Predominant topics are found to be sources of union, with topics centering around shared experiences as migrants struggling with visa regulations, as language learners, and as friends interested in sharing culture. The findings through BERTopic modeling as a tool of discourse analysis on large data shows a complex narrative creating distinctive concepts of migrants, divided into clustered groups to justify confrontational arguments and ensure the potential for union against the exploitative capitalist government policies is limited by the alienation from native workers, rather than a simple overall negative media narrative.  

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



