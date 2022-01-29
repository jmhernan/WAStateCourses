# WIP: LOAD TABLES TO POSTGRES
from enum import EnumMeta
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
import csv

this_file_path = os.path.abspath(os.getcwd()) # parse this out so that it works
project_root = os.path.split(this_file_path)[0]
raw_data_dir = os.path.join(project_root, 'data/raw-data/')

# raw_data_dir = '/Users/josehernandez/Documents/eScience/data/CCER_cadrs/ccer_data_10_2021/cadrs_collaboration_data_2021_10_05'
# WIP CHANGE NO TO CAPS!
raw_files = [f for f in os.listdir(raw_data_dir) if \
    os.path.isfile(os.path.join(raw_data_dir, f))] 

# WIP: Import dict with database info 
json_file = open(os.path.join(project_root,"data/db_credentials.json"))
db_cred= json.load(json_file)
json_file.close()

db_name = db_cred['db_name']
user = db_cred['user']
password = db_cred['password']

# WIP 
engine = create_engine(f'postgresql://{user}:{password}@localhost/{db_name}',
    echo=False)


# Need to make names postgres friendly 
def postgres_names(raw_name):
    pattern_1 = re.compile(r'(.)([A-Z][a-z]+)')
    pattern_2 = re.compile(r'([a-z0-9])([A-Z])')
    raw_name = re.sub('_', '', raw_name)
    name = pattern_1.sub(r'\1_\2', raw_name)
    lower_name = pattern_2.sub(r'\1_\2', name).lower() 
    table_name = lower_name.rsplit('.', 1)[0] 
    return table_name

# test reading in headers to make postgres compliant
header_dict = dict.fromkeys(raw_files, [])
for i, f in enumerate(raw_files):
    with open(os.path.join(raw_data_dir,f), 'r') as temp_f:
        reader = csv.reader(temp_f, delimiter='|')
        header = next(reader) 
        postgres_safe = [postgres_names(a) for a in header]
        header_dict[f] = postgres_safe

header_dict.keys()
#raw_files = [raw_files[0], raw_files[1]]
load_metadata = dict.fromkeys(raw_files, [])

for i,n in enumerate(raw_files):
    print(i,n)
    # Add Data
    t_load_0 = time.time()

    df = pd.read_csv(
        os.path.join(raw_data_dir, n),
        sep='|',
        quoting=csv.QUOTE_NONE,
        encoding='utf-8')
    df.columns = header_dict.pop(n)

    t_load_1 = time.time()
    load_t = t_load_1-t_load_0

    index = df.index
    d_rows = len(index)

    t_db_write_0 = time.time()
    df.to_sql(
        postgres_names(n), 
        engine,
        index=False,
        if_exists="replace", 
        chunksize=100000 
    )
    t_db_write_1 = time.time()
    db_write_t = t_db_write_1 - t_db_write_0
    load_metadata[n] = [d_rows, load_t, db_write_t]

load_metadata
with open(os.path.join(root_dir,"data/load_meta.json"), "w") as outfile:
    json.dump(load_metadata, outfile)  

# WIP: test files and column names row errors
#raw_files_test = raw_files[0],raw_files[3]
for i, n in enumerate(raw_files):
    dataset_metadata = dict.fromkeys(raw_files, [])
    for i, n in enumerate(raw_files):
        t0 = time.time()
        with open(os.path.join(raw_data_dir,n), 'r') as temp_f:
            col_count = [ len(l.split("|")) for l in temp_f.readlines() ]
        rows = len(col_count)
        max_c = max(col_count)
        min_c = min(col_count)
        t1 = time.time()
        total_t = round(t1-t0)
        dataset_metadata[n]=[min_c,max_c, rows, total_t]

# For column process
dataset_metadata
with open(os.path.join(root_dir,"data/data_column_meta.json"), "w") as outfile:
    json.dump(dataset_metadata, outfile) 
