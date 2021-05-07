"""
WIP:Cluster Sequences 
=================
Cluster course sequences using Word2Vec model embeddings. You need to train 
model by running sequence_embeddings.py
"""
import numpy as np 

from gensim.models import Word2Vec
import nltk

from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn import metrics
from sklearn.decomposition import PCA

from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

import preprocessing as pp

this_file_path = os.path.abspath(__file__)
# this_file_path = '/home/ubuntu/source/WAStateCourses/seqcrs/data/sequence_embeddings.py'
project_root = os.path.split(os.path.split(this_file_path)[0])[0]
data_path = os.path.join(project_root, "data") + '/'
wastate_db = data_path + 'ccer_data.db'
sql_script = project_root + '/seqcrs/data/course_history_load.sql'

course_df = pp.execute_sql(sql_filename=sql_script, db_path=wastate_db)
df_cls = course_df[1:1000]

columns = ['ResearchID', 'CourseTitle']
pivot_df = df_cls[columns]

def pivot_fun(df):
    reshaped_df = df.groupby('ResearchID').agg({'CourseTitle':lambda x: list(x)}).reset_index()
    return(reshaped_df)

course_list = pp.parallelize_df(pivot_df, pivot_fun)

# very slow inspect try with reduced sample
course_seq_ls = course_list['CourseTitle'].to_list()

course_seq = pp.clean_courses(course_seq_ls)

# Word embedding model
# Load model trained on universe of transcripts
model_baseline = Word2Vec(course_seq, min_count=1) 
model_baseline.wv.__getitem__('schsurvivskills')
np.mean(model_baseline.wv.__getitem__('schsurvivskills'), axis=0)

# Create course sequence embedding aggregation vectorizer
# 1. Each course receives embedding value from model.
# 2. They are aggregated and averaged.

def vectorize_embed(vec_sequence, w2v_model):
    """
    vec_sequence: list of vectorized course sequences
    w2v_model: trained Word2Vec model to extract embeddings 
    """
    embedding = []
    n_word = 0
    for course in vec_sequence:
        try:
            if n_word == 0:
                embedding = w2v_model.wv.__getitem__(course)
            else:
                embedding = w2v_model.wv.__getitem__(course)
            n_word+=1
        except:
            pass
        return np.asarray(embedding)/n_word

l = []
for i in course_seq:
    l.append(vectorize_embed(i, model_baseline))

X = np.array(l).reshape(-1,1)

k_matrix = []

for i in range(1,4):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    k_matrix.append(kmeans.inertia_)

plt.plot(k_matrix)

n_cluster = 2

clf = KMeans(n_clusters=n_cluster,
    max_iter=100,
    init='k-means++',
    n_init=1)

labels = clf.fit_predict(X)

for index, course_seq in enumerate(course_seq):
    print(str(labels[index])+":"+ str(course_seq))