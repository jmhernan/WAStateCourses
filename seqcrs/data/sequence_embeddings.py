# Sequence summary functions and tools
# leverage Sequence Graph Transform embeddings 
# Compare to word2vec

import pandas as pd
from sqlalchemy import create_engine

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(os.path.split(os.path.split(this_file_path)[0])[0])[0]

data_path = os.path.join(project_root, "data") + '/'

wastate_db = data_path + 'ccer_data.db'

# WIP: Function to connect to db and load data
engine = create_engine(f"sqlite:///{wastate_db}", echo=True)
sqlite_conn = engine.connect()

model_df = pd.read_sql_table(
    'sequence_processed',
    con=sqlite_conn
)

sqlite_conn.close()
