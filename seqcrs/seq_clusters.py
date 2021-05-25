"""
WIP:Cluster Sequences 
=================
Cluster course sequences using Word2Vec model embeddings. You need to train 
model by running sequence_embeddings.py
"""
import numpy as np 

from gensim.models import Word2Vec
import nltk

import pandas as pd

from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score

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
df_cls = course_df[1:500000]

columns = ['ResearchID', 'CourseTitle']
pivot_df = df_cls[columns]

def pivot_fun(df):
    reshaped_df = df.groupby('ResearchID').agg({'CourseTitle':lambda x: list(x)}).reset_index()
    return(reshaped_df)

course_list = pp.parallelize_df(pivot_df, pivot_fun)

# very slow inspect try with reduced sample
course_seq_ls = course_list['CourseTitle'].to_list()

course_seq = pp.clean_courses(course_seq_ls)
len(course_seq)
# Word embedding model
# Load model trained on universe of transcripts
model_baseline = Word2Vec(course_seq, min_count=1) 
# look at model outputs
model_baseline.wv.__getitem__('schsurvivskills')
model_baseline.wv.most_similar('calculus')

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

# Try alternative
def vectorize_courses(list_of_courses, w2v_model):
    """Generate vectors for list of courses using a word embedding

    Args:
        list_of_courses: List of course sequences
        w2v_model: Trained word embedding model

    Returns:
        List of course vectors
    """
    features = []

    for tokens in list_of_courses:
        zero_vector = np.zeros(w2v_model.vector_size)
        vectors = []
        for token in tokens:
            if token in w2v_model.wv:
                try:
                    vectors.append(w2v_model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features
    
vectorized_courses = vectorize_courses(course_seq, w2v_model=model_baseline)
len(vectorized_courses), len(vectorized_courses[0])

# def mini batch k-means for this 
def mini_kmeans(X, k, mb, print_silhouette):
    """Generate clusters and print Silhouette metrics using scikit MBKmeans

    Args:
        X: Matrix of features
        k: number of clusters
        mb: Size of mini-batches
        print_silhouette: Print per cluster

    Returns: 
        Trained clustering model and labels based on X.
    """
    km = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(X)
    print(f"For n_clusters = {k}")
    print(f"Silhouette coefficient: {silhouette_score(X, km.labels_):0.2f}")
    print(f"Inertia:{km.inertia_}")

    if print_silhouette:
        sample_silhouette_values = silhouette_samples(X, km.labels_)
        print(f"Silhouette values:")
        silhouette_values = []
        for i in range(k):
            cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
            silhouette_values.append(
                (
                    i,
                    cluster_silhouette_values.shape[0],
                    cluster_silhouette_values.mean(),
                    cluster_silhouette_values.min(),
                    cluster_silhouette_values.max(),
                )
            )
        silhouette_values = sorted(
            silhouette_values, key=lambda tup: tup[2], reverse=True
        )
        for s in silhouette_values:
            print(
                f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
            )
    return km, km.labels_

# run the mini batch KMeans 
# WIP: HOW MANY CLUSTERS? 
clustering, cluster_labels = mini_kmeans(
	X=vectorized_courses,
    k=40,
    mb=300,
    print_silhouette=True,
)

df_clusters = pd.DataFrame({
    "text": course_seq_ls,
    "tokens": [" ".join(text) for text in course_seq],
    "cluster": cluster_labels
})

# LOOKS AT THE COURSE NAMES INDIVIDUALLY
print("Most representative terms per cluster (based on centroids):")
for i in range(40):
    tokens_per_cluster = ""
    most_representative = model_baseline.wv.most_similar(positive=[clustering.cluster_centers_[i]], topn=5)
    for t in most_representative:
        tokens_per_cluster += f"{t[0]} "
    print(f"Cluster {i}: {tokens_per_cluster}")

# LOOKS AT THE COURSE SEQUENCES PER CLUSTER INDIVIDUALLY
test_cluster = 29
most_representative_docs = np.argsort(
    np.linalg.norm(vectorized_courses - clustering.cluster_centers_[test_cluster], axis=1)
)
for d in most_representative_docs[:5]:
    print(course_seq_ls[d])
    print("-------------")