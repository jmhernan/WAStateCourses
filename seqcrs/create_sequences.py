#!/usr/bin/env python
import numpy as np
import pandas as pd
import json#
import os
import re
import sys
from pathlib import Path
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer#

import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.types import NVARCHAR, Integer, Text

import time

this_file_path = os.path.abspath(__file__)

preprocess_path = os.path.split(os.path.split(this_file_path)[0])[0]
sys.path.insert(1, preprocess_path)

project_root = os.path.split(os.path.split(this_file_path)[0])[0]

import preprocessing as pp

data_path = os.path.join(project_root, "data") + '/'

# Get columns of interests WIP: Check query!!
query_txt = '''
SELECT * 
FROM ghf_tukwila
WHERE ResearchID IN (SELECT ResearchID 
    FROM complete_hs_records WHERE DistinctGradeLevelCount = 4);
'''
print(query_txt)

wastate_db = data_path + 'ccer_data.db'

# connect to db and load data
df_courses = pp.execute_sql(db_path=wastate_db, sql_txt=query_txt)

df_courses.shape
df_courses.columns

df_courses['GradeLevelWhenCourseTaken'] = df_courses['GradeLevelWhenCourseTaken'].astype(int)
# Create data file of ordered course sequences for cohort 
# 1. Order by term 
# 2. Omit failed courses 
# 3. pivot wide for sequence by Research ID 
# 4. Convert to list

# Sort
# WIP: Sorts student course history by grade (9,10,11,12) 
# and course alpha order
df_sorted = df_courses.groupby(
        ['ResearchID','GradeLevelWhenCourseTaken'], 
        sort=True).apply(lambda x: x.sort_values(
        ['GradeLevelWhenCourseTaken','CourseTitle'], 
        ascending = [False,True])).reset_index(drop=True)
##################

failed_courses = df_sorted['CreditsEarned'].astype(float) > 0 
df_passed_crs =  df_sorted[failed_courses].reset_index(drop=True)

columns = ['ResearchID', 'CourseTitle']
pivot_df = df_passed_crs[columns]

# WIP: Most promising method so far one row per student and course sequence
course_list = pivot_df.groupby('ResearchID').agg({'CourseTitle':lambda x: list(x)}).reset_index()

# ADD codes for known CADRS 
tukwila_coded_fn = 'tukwila_aggregated_results.csv'
cadrs_tukwila =  pd.read_csv(os.path.join(data_path,tukwila_coded_fn),
                             delimiter = ',')

cadrs_tukwila['cadr_sum'] = cadrs_tukwila['art_cadr_v'] + \
    cadrs_tukwila['math_cadr_v'] + \
    cadrs_tukwila['eng_cadr_v'] + \
    cadrs_tukwila['sci_cadr_v'] + \
    cadrs_tukwila['soc_cadr_v'] + \
    cadrs_tukwila['flang_cadr_v']-5

cadrs_tukwila_sub = cadrs_tukwila[['ResearchID','cadr_sum']].dropna()
cadrs_tukwila_sub.shape

additional_ids = pp.get_metadata_dict(os.path.join(data_path, "tukwila_handcoded.json"))

additional_cadrs = pd.DataFrame(additional_ids)

# Union of cadr eligle students 
outcomes_y1 = pd.concat([cadrs_tukwila_sub, additional_cadrs], ignore_index=True).drop_duplicates().reset_index(drop=True)
outcomes_y1.dtypes

# Check coverage
sum(cadrs_tukwila_sub['ResearchID'].isin(course_list['ResearchID']))
sum(additional_cadrs['ResearchID'].isin(course_list['ResearchID']))

sum(outcomes_y1['ResearchID'].isin(course_list['ResearchID'])) # issue here prob diff data types force to str

sum(additional_cadrs['ResearchID'].isin(cadrs_tukwila_sub['ResearchID']))

# Join the course student table
course_list['ResearchID'] = course_list['ResearchID'].astype(str) 
outcomes_y1['ResearchID'] = outcomes_y1['ResearchID'].astype(str)

results_df = pd.merge(course_list, outcomes_y1, on = 'ResearchID', how = 'left')

results_df['cadr_sum'] = results_df['cadr_sum'].fillna(0)

sum(results_df['cadr_sum'])

# Clean up list of sequences
# 1. Replace spaces + / with underscores
# 2. Lower Case
course_seq_ls = course_list['CourseTitle'].to_list() 

course_seq = pp.clean_courses(course_seq_ls, tokenize=False)

# Check sequence distribution for Tukwila  
course_cnt = pp.get_top_n_courses(course_seq, n=100)

type(course_cnt)

course_name = list(zip(*course_cnt))[0]
cnt = list(zip(*course_cnt))[1]
x_pos = np.arange(len(course_cnt)) 

plt.bar(x_pos, cnt, align='center')
plt.xticks(x_pos, course_name) 
plt.ylabel('Course Counts')
plt.xticks(rotation=90, size=3)
plt.show()

# Course counts per Student
n_courses = [len(element) for element in course_seq_ls]
len(course_seq_ls[24])
n_courses.sort(reverse=True)

x_pos = np.arange(len(n_courses)) 

plt.bar(x_pos, n_courses, align='center')
plt.xticks(x_pos) 
plt.ylabel('Course Counts')
plt.show()

# WIP: WE CAN SEE THAT NOT EVERYONE HAS COMPLETE RECORDS SOME STUDENTS HAVE A 
# TOTAL OF 10 COURSES OVERALL
# NEED TO DO A SUBSET...A COURSE IN ALL 4 YEARS OR SIMILAR

# Save to SQL DB
results_df['course_seq'] = course_seq
df_sql = results_df.drop(['CourseTitle'], axis=1)
# save 
from sqlalchemy import create_engine
from sqlalchemy.types import NVARCHAR, Integer, Text

engine = create_engine(f"sqlite:///{wastate_db}", echo=True)
sqlite_connection = engine.connect()

sql_table = "sequence_processed"
df_sql.to_sql(sql_table, sqlite_connection, if_exists='replace',
    dtype = {'ResearchID':NVARCHAR(), 'cadr_sum':Integer(), 'course_seq':NVARCHAR()})

engine.execute("SELECT * FROM sequence_processed limit 10").fetchall()

sqlite_connection.close()