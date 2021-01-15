#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import re
import sys
from pathlib import Path
import matplotlib.pyplot as plt

import gensim
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

from nltk.tokenize import word_tokenize

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(this_file_path)[0]

import text_preprocess as tp

data_path = project_root + '/data/'

# data
crs_cat_df = pd.read_csv(os.path.join(data_path, '2019-20StateCourseCodes.csv'), skiprows = 3, delimiter = ',')

text = crs_cat_df['Description'].astype(str)

clean_text = text.apply(tp.clean_text)

clean_text = tp.remove_non_ascii(clean_text)

tp.get_top_n_words(clean_text, n=100)

corpus = [word for word in clean_text if word not in stop_words]

tokenized_text = [word_tokenize(i) for i in corpus]

# word embedding model
model_baseline = Word2Vec(tokenized_text, min_count=1) 

len(list(model_baseline.wv.vocab))

# 100 most occuring 
model_baseline.wv.index2entity[:100] # same as top_n words

# Build a function
keywords = [
    'algebra',
    'calculus'
]

# gets combination 
tp.get_similar_words(keywords, 10, model_baseline)

model_baseline.most_similar(positive='algebra', topn=10)
model_baseline.most_similar(positive='geometry', topn=10)
model_baseline.most_similar(positive='science', topn=10)
model_baseline.most_similar(positive='psychology', topn=10)

Gmodel = gensim.models.KeyedVectors.load_word2vec_format('/Users/josehernandez/Documents/eScience/GoogleNews-vectors-negative300.bin', binary=True)

tp.get_similar_words(keywords, 10, Gmodel)

Gmodel.most_similar(positive='algebra', topn=10)

tp.get_similar_words(keywords, 10, Gmodel)