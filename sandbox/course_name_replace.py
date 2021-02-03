# If name is less than 5 characters replace with State course
# see names with less than 5 characters 
import numpy as np
import pandas as pd
import os
import re
import sys
from pathlib import Path
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(os.path.split(this_file_path)[0])[0]
 
data_path = os.path.join(project_root, "data") + '/'

db = data_path + 'ccer_data.db'

con = sqlite3.connect(db)

# Get columns of interests
query_txt = "SELECT * FROM ghf_renton;"
print(query_txt)

df_courses = pd.read_sql_query(query_txt, con)
con.close()

# Isolate district course info + state course info
# 1. Get index of courses that are > 5 characters 
# 2. Check if state course name is in a better shape
# 3. replace CourseTitle courses with the state names 
# 4. Feed back to te main data processing script
columns = ['ReportSchoolYear','CourseID','CourseTitle','StateCourseCode', 'StateCourseName']
df_course_cleanup = df_courses[columns]
df_course_cleanup['char_len'] = df_course_cleanup['CourseTitle'].str.len()

