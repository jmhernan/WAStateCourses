# WIP: LOAD TABLES TO POSTGRES
import os
from posixpath import splitext
import sys
from pathlib import Path
import json
import re
from matplotlib.pyplot import table
import time

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import base
from sqlalchemy.types import NVARCHAR, Integer, Text

root_dir = os.path.abspath(os.getcwd())
raw_data_dir = os.path.join(root_dir, 'data/ccer_data_10_2021/cadrs_collaboration_data_2021_10_05/')
project_root = os.path.join(root_dir, 'source/WAStateCourses')

# WIP CHANGE NO TO CAPS!
raw_files = [f for f in os.listdir(raw_data_dir) if os.path.isfile(os.path.join(raw_data_dir, f))] 

# WIP: Import dict with database info 
json_file = open(os.path.join(project_root,"data/db_credentials.json"))
db_cred= json.load(json_file)
json_file.close()

db_name = db_cred['db_name']
user = db_cred['user']
password = db_cred['password']

# WIP 
engine = create_engine(f'postgresql://{user}:{password}@localhost/{db_name}', echo=False) # Find a way to use this in messages

# Add Data
t0 = time.time()
df = pd.read_csv(
    os.path.join(raw_data_dir, raw_files[4]),
    sep='|'
)
t1 = time.time()
t1-t0

index = df.index
len(index)
# Need to make nae postgres friendly 

def postgres_names(raw_name):
    pattern_1 = re.compile(r'(.)([A-Z][a-z]+)')
    pattern_2 = re.compile(r'([a-z0-9])([A-Z])')
    raw_name = re.sub('_', '', raw_name)
    name = pattern_1.sub(r'\1_\2', raw_name)
    lower_name = pattern_2.sub(r'\1_\2', name).lower() 
    table_name = lower_name.rsplit('.', 1)[0] 
    return table_name

t0 = time.time()
df.to_sql(
    postgres_names(raw_files[4]), 
    engine,
    index=False,
    if_exists="replace", 
    chunksize=100000 
)
t1 = time.time()
t1-t0