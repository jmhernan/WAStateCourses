#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import re
import sys
from pathlib import Path
import sqlite3

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(this_file_path)[0]
 
data_path = os.path.join(project_root, "data") + '/'

db = data_path + 'ccer_data.db'

con = sqlite3.connect(db)

# Get columns of interests
query_txt = "SELECT * FROM ghf_renton;"
print(query_txt)

df_courses = pd.read_sql_query(query_txt, con)
con.close()

df_courses.shape
df_courses.columns


# Create data file of ordered course sequences for cohort 
# ResearchID,ReportSchoolYear,DistrictName,TermEndDate,Term,
# GradeLevelWhenCourseTaken,CourseID,CourseTitle,CreditsEarned,
# StateCourseCode,StateCourseName,dSchoolYear
single_case = df_courses['ResearchID'] == '#####'
single_case_df = df_courses[single_case]
# 1. Order by term 
# 2. ommit failed courses 
# 3. pivot wide for sequence by Research ID 
# 4. Convert to list
df_sorted = df_courses.sort_values(['CourseTitle'], ascending=True).groupby(['ResearchID'], sort=False)\
    .apply(lambda x: x.sort_values(['TermEndDate'], ascending = True)).reset_index(drop=True)

double_case = ['#######', '#######']
double_case_df = df_sorted[df_sorted.ResearchID.isin(double_case)] 

failed_courses = double_case_df['CreditsEarned'].astype(float) > 0 
double_case_df =  double_case_df[failed_courses].reset_index(drop=True)

columns = ['ResearchID', 'CourseTitle']
pivot_df = double_case_df[columns]

test = pivot_df.groupby('ResearchID')['CourseTitle'].apply(lambda df: df.reset_index(drop=True)).unstack()

# Most promising method so far!
pivot_df.groupby('ResearchID').agg({'CourseTitle':lambda x: list(x)})
# Clean up list of sequences
# 1. Replace spaces + / with underscores
# 2. Lower Case

# Check sequence distribution for Renton  
