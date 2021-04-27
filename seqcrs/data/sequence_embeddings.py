# Sequence summary functions and tools
# leverage Sequence Graph Transform embeddings 
# Compare to word2vec
# Train on sequences of all courses taken in RMP Region

import pandas as pd
from sqlalchemy import create_engine

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(os.path.split(os.path.split(this_file_path)[0])[0])[0]

data_path = os.path.join(project_root, "data") + '/'

wastate_db = data_path + 'ccer_data.db'

# WIP: Function to connect to db and load data
engine = create_engine(f"sqlite:///{wastate_db}", echo=True)
sqlite_conn = engine.connect()

course_df = pd.read_sql_table(
    'hsCourses',
    con=sqlite_conn
)

sqlite_conn.close()

course_df.columns
course_df['ReportSchoolYear'].value_counts()
len(course_df['ResearchID'].unique().tolist())
# 125,000 course sequences

# Prep sequences for Word2Vec 
