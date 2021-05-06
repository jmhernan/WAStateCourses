# Sequence summary functions and tools
# leverage Sequence Graph Transform embeddings 
# Compare to word2vec
# Train on sequences of all courses taken in RMP Region
import os
import re
import sys

import pandas as pd
from sqlalchemy import create_engine

this_file_path = os.path.abspath(__file__)
this_file_path = '/home/ubuntu/source/WAStateCourses/seqcrs/data/sequence_embeddings.py'
project_root = os.path.split(os.path.split(os.path.split(this_file_path)[0])[0])[0]

import preprocessing as pp

from nltk.tokenize import word_tokenize

import smart_open
smart_open.open = smart_open.smart_open
from gensim.models import Word2Vec

data_path = os.path.join(project_root, "data") + '/'

wastate_db = data_path + 'ccer_data.db'

sql_script = project_root + '/seqcrs/data/course_history_load.sql'

course_df = pp.execute_sql(sql_filename=sql_script, db_path=wastate_db)

course_df.columns
len(course_df['ResearchID'].unique().tolist()) # ~125,000 course sequences

# Prep sequences for Word2Vec 
columns = ['ResearchID', 'CourseTitle']
pivot_df = course_df[columns]

# WIP: Most promising method so far one row per student and course sequence
# NOT EFFICIENT WITH LARGER FILE
# Try multiprocess 
# TRY SQL
# No PIVOT function in sqlite.
def pivot_fun(df):
    reshaped_df = df.groupby('ResearchID').agg({'CourseTitle':lambda x: list(x)}).reset_index()
    return(reshaped_df)

course_list = pp.parallelize_df(pivot_df, pivot_fun)

# very slow inspect try with reduced sample
course_seq_ls = course_list['CourseTitle'].to_list()

course_seq = pp.clean_courses(course_seq_ls)

# word embedding model
model_baseline = Word2Vec(course_seq) 

model_baseline.save('course_baseline_model.bin')

len(list(model_baseline.wv.vocab))

# 100 most occuring 
model_baseline.wv.index2entity[:500] # same as top_n words

# Build a function
keywords = [
    'career_choices'
]

def get_similar_words(list_words, top, wb_model):
    list_out = wb_model.wv.most_similar(list_words, topn=top)
    return list_out

# gets combination 
get_similar_words(keywords, 20, model_baseline)