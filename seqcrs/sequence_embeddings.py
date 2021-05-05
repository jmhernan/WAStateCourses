# Sequence summary functions and tools
# leverage Sequence Graph Transform embeddings 
# Compare to word2vec
# Train on sequences of all courses taken in RMP Region
import pandas as pd
from sqlalchemy import create_engine

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(os.path.split(this_file_path)[0])[0]

preprocess_path = os.path.split(this_file_path)[0]
sys.path.insert(1, preprocess_path)

import preprocessing as pp

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

# very slow inspect
course_seq_ls = course_list['CourseTitle'].to_list()

course_seq = pp.clean_courses(course_seq_ls)

# Word2Vec
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