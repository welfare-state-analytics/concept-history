# 1. create lemmas over time function
# 2. create senses by party function
# 3. create add born/age fuctions
# 4. cluster/use pca to plot in 2 dimensions,

import numpy as np
import os, json
from pygibbs import utils
import matplotlib.pyplot as plt
import pandas as pd
import random
from collections import Counter
from bidict import bidict
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import seaborn as sns
from progressbar import progressbar
from time import time
def cleanYear(year):
    if len(year) == 4:
        return int(year)
    return int(year[:4])

folder = 'results/second-run/j_window_5_topic_5'
meta = pd.read_csv('data/context_windows/j_meta.csv')

in_path = os.path.join(folder, 'model')
Nd = np.load(os.path.join(in_path, 'Nd.npy'))
Nk = np.load(os.path.join(in_path, 'Nk.npy'))
theta = np.load(os.path.join(in_path, 'theta.npy'))
phi = np.load(os.path.join(in_path, 'phi.npy'))
z = np.load(os.path.join(in_path, 'z.npy'))
logdensity = np.load(os.path.join(in_path, 'logdensity.npy'))
posterior = np.load(os.path.join(in_path, 'posterior.npy'))

file = '_'.join(folder.split('/')[-1].split('_')[:3]) + '.json'

with open(os.path.join('data/context_windows', file)) as f:
    data = json.load(f)
with open(os.path.join(in_path, 'vocab.json')) as f:
    vocab = json.load(f)
with open('data/stopwords-sv.txt','r') as f:
    stopwords = f.read().splitlines()

doc = data["doc"]
w = data["w"]
file_dir = data["file_path"]
years = list(map(lambda x: cleanYear(x.split('/')[3]), file_dir))
target = list(map(lambda x: x[:4], data["target"])) # Lemmatize
M, V, K = len(z), np.shape(Nk)[1], len(Nd)
z, phi, Nk = utils.z_sorter(z, phi, Nk, K)

color = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1',\
         '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF'][:K]

def topDocs(doc, w, posterior, K, file_dir, n=10, seed=123):
    """
    Computes the top n p(m|K=k)
    Retrives documents most associated with each topic.
    Shuffling is used to solve conflicts with equally dominant documents.
    """
    random.seed(seed)
    M = len(posterior)
    indices_map = random.sample(range(M), M)
    indices_map = np.array([list(range(M)), indices_map]).T
    topdocs = []

    for k in range(K):
        # Get original topdoc indices from shuffled posterior
        indices = (-posterior[indices_map[:, 1], k]).argsort()[:n]
        indices = indices_map[indices, 0]
        topdocs.append(f'Topic {k}:')

        for idx in indices:
            word_indices = [i for i,m in enumerate(doc) if m == idx]
            words = ' '.join(w[word_indices[0]:word_indices[-1]+1])            
            topdocs.append(f'{words} ({file_dir[doc[word_indices[0]]]})')
    
    return topdocs

years = list(map(lambda x: x.split('/')[3], file_dir))
print(sorted(set(years)))






#x = topDocs(doc, w, posterior, K, file_dir, n=10, seed=1234)
#print(x)



