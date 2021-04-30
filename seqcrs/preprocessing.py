import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import json
import pandas as pd
from sqlalchemy import create_engine

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

# WIP: 
# 1. Load raw data tables ✅
# 2. Load using SQL script ✅
# 3. Load overriding with your own script
# 4. Provide option for summary stats and viz.
def load_sql_table(table_name, db_name):
    engine = create_engine(f"sqlite:///{db_name}", echo=False) # Find a way to use this in messages
    sqlite_conn = engine.connect()
    df = pd.read_sql_table(
        table_name,
        con=sqlite_conn
    )
    sqlite_conn.close()
    return df

def execute_sql(db_path,sql_filename=None,sql_txt=None):
    # Connect to DB
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    sqlite_conn = engine.connect()

    if sql_filename is not None:
        # Open and read the file as a single buffer
        fd = open(sql_filename, 'r')
        query_txt = fd.read()
        fd.close()
    else:
        query_txt = sql_txt
    
    # Execute and load table as pandas df
    df = pd.read_sql(query_txt, con=sqlite_conn)
    # Close the connection
    sqlite_conn.close()
    return df

# WIP: create sequences
# provide option for summary stats and viz
def to_sequence():
    pass

# WIP: save to sql db
def save_to_db():
    pass
