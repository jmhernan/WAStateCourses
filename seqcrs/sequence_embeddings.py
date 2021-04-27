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
fd = open(project_root + '/seqcrs/data/course_history_load.sql', 'r')
sqlFile = fd.read()
fd.close()

def exec_sql_from_file(filename, db_path):
    # Open and read the file as a single buffer
    fd = open(filename, 'r')
    query_txt = fd.read()
    fd.close()
    # Connect to DB
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    sqlite_conn = engine.connect()
    # Execute and load table as pandas df
    df = pd.read_sql(query_txt, con=sqlite_conn)
    # Close the connection
    sqlite_conn.close()
    return df

sql_script = project_root + '/seqcrs/data/course_history_load.sql'

test = exec_sql_from_file(filename=sql_script, db_path=wastate_db)

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
