# Embedding data prep to export and 
# visualize using Tensorboard Projections
import io
import os
import re
import shutil
import string

import smart_open
smart_open.open = smart_open.smart_open
from gensim.models import Word2Vec
from gensim.test.utils import datapath

import tensorflow as tf

# Word embeddings 
embeddings_aws = '/home/ubuntu/source/WAStateCourses/seqcrs/course_baseline_model.bin'

#WIP double underscore, periods
model = Word2Vec.load(datapath(embeddings_test))
model.wv.most_similar('geometry', topn=10) 
model.wv.__getitem__('algebra_1')

# Extract model components
weights = model.wv.vectors
words = list(model.wv.vocab)

# save to use embeddings projector 
out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(words):
    if index == 0:
        continue # skip 0, it's padding
    vec = weights[index]
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_m.write(word + "\n")
out_v.close()
out_m.close()
