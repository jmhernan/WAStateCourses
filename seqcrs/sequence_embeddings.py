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

course_df = pp.exec_sql_from_file(filename=sql_script, db_path=wastate_db)

course_df.columns
len(course_df['ResearchID'].unique().tolist())
# ~125,000 course sequences

# Prep sequences for Word2Vec 
