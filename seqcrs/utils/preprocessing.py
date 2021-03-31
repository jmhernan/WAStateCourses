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

    def clean_courses(text=list):
    crs_ls = [[x.strip() for x in l] for l in course_seq_ls] 
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