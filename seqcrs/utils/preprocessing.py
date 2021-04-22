import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import json

def get_top_n_courses(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    """
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def get_metadata_dict(metadata_file):
    metadata_handle = open(metadata_file)
    metadata = json.loads(metadata_handle.read())
    return metadata

def clean_courses(text_ls):
    crs_ls = [[x.strip() for x in l] for l in text_ls] 
    crs_ls = [[x.replace('(', '') for x in l] for l in crs_ls]
    crs_ls = [[x.replace(')', '') for x in l] for l in crs_ls]
    crs_ls = [[x.replace(':', '') for x in l] for l in crs_ls]
    crs_ls = [[x.lower() for x in l] for l in crs_ls]
    crs_ls = [[x.replace('/', '_') for x in l] for l in crs_ls]
    crs_ls = [[x.replace('-', '') for x in l] for l in crs_ls]
    crs_ls = [[x.replace(' ', '_') for x in l] for l in crs_ls]
    crs_ls = [[" ".join(w for w in l)] for l in crs_ls]
    crs_ls = [x for sublist in crs_ls for x in sublist] 
    return crs_ls

# WIP: Load raw data 
# provide option for summary stats and viz
def load_sql():
    pass

# WIP: create sequences
# provide option for summary stats and viz
def to_sequence():
    pass

# WIP: save to sql db
def save_to_db():
    pass
