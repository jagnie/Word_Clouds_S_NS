
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import time
from wordcloud import WordCloud
from collections import Counter
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

#loading all necessary libraries
import numpy as np
import pandas as pd

import string
import collections
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.cm as cm
import matplotlib.pyplot as plt

df = pd.read_csv('enron2_clean_wc.csv',sep=',')  
pd.set_option('display.max_colwidth',50)
pd.set_option('display.max_columns',30)
print(df.head())
df.drop_duplicates(inplace = True)





spam_words = ' '.join(list(map(str,df[df['label'] == 1 ]['clean_emails'])))
#spam_words = ' '.join(list(map(str,df['clean_emails'])))
timestr = time.strftime("%Y%m%d-%H%M%S")
wordcloud_spam = WordCloud(background_color="white",width=1600, height=800, max_words=500).generate(spam_words)
wordcloud_spam.to_file("wordcloud"+timestr+".png")
plt.figure(figsize = (20,20))
plt.imshow(wordcloud_spam, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


filtered_words_spam = [word for word in spam_words.split()]
counted_words_spam = collections.Counter(filtered_words_spam)

word_count_spam = {}

for letter, count in counted_words_spam.most_common(50):
    word_count_spam[letter] = count
    
for i,j in word_count_spam.items():
        print('Word: {0}, count: {1}'.format(i,j))
