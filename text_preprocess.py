# Functions specific to preprocess raw extract data from GoogleSheets

import re
import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# regex conditions for text cleanup
BAD_SYMBOLS_RE = re.compile(r'[\W]')
REM_LETTER = re.compile(r'(\b\w{1}\b)')
REM_NAN_WRD = re.compile(r'^(?:\s*nan\s*)')

# handle json
def get_metadata_dict(metadata_file):
    metadata_handle = open(metadata_file)
    metadata = json.loads(metadata_handle.read())
    return metadata

# clean text 
def clean_text(text):    
    text = BAD_SYMBOLS_RE.sub(' ', text)
    text = REM_LETTER.sub('', text) 
    text = REM_NAN_WRD.sub('', text)
    return text

def update_abb(text, json_abb):
    abb_cleanup = {r'\b{}\b'.format(k):v for k, v in json_abb.items()}
    abb_replace = text.replace(to_replace =abb_cleanup, regex=True)
    return abb_replace

def get_top_n_words(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    """
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def remove_non_ascii(txt):
    clean_txt = [0]*len(txt)
    for ind, t in enumerate(txt):
        t = t.encode('ascii', 'ignore')
        clean_txt[ind] = t.decode('ascii')
    return clean_txt

def get_similar_words(list_words, top, wb_model):
    list_out = list_words
    for w in wb_model.most_similar(list_words, topn=top):
        list_out.append(w[0])
    return list(set(list_out))