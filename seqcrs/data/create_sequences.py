#!/usr/bin/env python
import numpy as np
import pandas as pd
import os
import re
import sys
from pathlib import Path
import sqlite3
from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt

this_file_path = os.path.abspath(__file__)
project_root = os.path.split(os.path.split(os.path.split(this_file_path)[0])[0])[0]
 
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
# 1. Order by term 
# 2. ommit failed courses 
# 3. pivot wide for sequence by Research ID 
# 4. Convert to list
df_sorted = df_courses.sort_values(['CourseTitle'], ascending=True).groupby(['ResearchID'], sort=False)\
    .apply(lambda x: x.sort_values(['TermEndDate'], ascending = True)).reset_index(drop=True)

failed_courses = df_sorted['CreditsEarned'].astype(float) > 0 
df_passed_crs =  df_sorted[failed_courses].reset_index(drop=True)

columns = ['ResearchID', 'CourseTitle']
pivot_df = df_passed_crs[columns]

# Most promising method so far!
course_lists = pivot_df.groupby('ResearchID').agg({'CourseTitle':lambda x: list(x)})

# Clean up list of sequences
# 1. Replace spaces + / with underscores
# 2. Lower Case
course_seq_ls = course_lists['CourseTitle'].to_list() 
test = [['IB Language A: Language and Literature HL','IB History of the Americas HL'],['(H) CHEMISTRY', 'IB Language A: Language and Literature HL']]

# WIP: make function replace parentheses and others
test = [[x.strip() for x in l] for l in course_seq_ls] 
test = [[x.replace('(', '') for x in l] for l in test]
test = [[x.replace(')', '') for x in l] for l in test]
test = [[x.replace(':', '') for x in l] for l in test]
test = [[x.lower() for x in l] for l in test]
test = [[x.replace('/', '_') for x in l] for l in test]
test = [[x.replace(' ', '_') for x in l] for l in test]

test = [[" ".join(w for w in l)] for l in test]
test = [x for sublist in test for x in sublist] # courses as sequences flattened lists

# Check sequence distribution for Renton  
def get_top_n_courses(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    """
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

course_cnt = get_top_n_courses(test, n=100)
type(course_cnt)

course_name = list(zip(*course_cnt))[0]
cnt = list(zip(*course_cnt))[1]
x_pos = np.arange(len(course_cnt)) 

plt.bar(x_pos, cnt, align='center')
plt.xticks(x_pos, course_name) 
plt.ylabel('Course Counts')
plt.xticks(rotation=90, size=3)
plt.show()

# Course counts
n_courses = [len(element) for element in course_seq_ls]
len(course_seq_ls[24])
n_courses.sort(reverse=True)

x_pos = np.arange(len(n_courses)) 

plt.bar(x_pos, n_courses, align='center')
plt.xticks(x_pos) 
plt.ylabel('Course Counts')
plt.show()