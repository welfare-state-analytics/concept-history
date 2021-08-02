import numpy as np
import pandas as pd
import os,json
from collections import Counter
from scipy.special import loggamma
from progressbar import progressbar

# Location of test_data
path = 'tests/test_data'
files = ['test_stan_data.json','test_theta.json','test_phi.json']

# Load files
lst = []
for i in range(len(files)):
    with open(path+'/'+files[i]) as f:
        lst.append(json.load(f))

data,theta_true,phi_true = lst
theta_true = np.array((1,2,3,4)) / (1+2+3+4) # Bug in git data, enter manually until fixed
phi_true = np.array(phi_true)
K,V,M,N,alpha,beta,z_true,w,doc = list(data.values())

# Create df, make index go from 0
z_true = [x-1 for x in z_true] # Index from 0
df = pd.DataFrame(np.array((doc,w)).T) - 1
df.columns = ["doc", "w"]
df = df.groupby('doc')['w'].apply(list).reset_index()

# Make these more general later
df["z"] = np.random.choice(K, M)
df["n"] = [int(N/M)]*M

# Create sufficient statistic matrices
Nd = np.zeros(K, dtype = 'int')
Nk = np.zeros((K,V), dtype = 'int')

for doc in range(len(df)):
    d,w,z,n = df.loc[doc]
    Nd[z] += 1
    np.add.at(Nk[z], w, 1)

# Initiate theta,phi as unbiased sample estimates
theta = Nd / Nd.sum()
phi = Nk / Nk.sum(axis=1)[:, np.newaxis]

# Run algo
epochs = 50

for epoch in progressbar(range(epochs)):
    for doc in range(len(df)):
        d,w,z,n = df.loc[doc]

        # Get word counts for document
        w_count = Counter(w)
        counts = list(w_count.values())
        keys = list(w_count.keys())

        # Decrement counts
        np.subtract.at(Nk[z], keys, counts)
        Nd[z] -= 1

        # Sample sense
        probs = theta.copy()
        for i in w:
            probs *= phi[:,i]
        probs = probs / probs.sum()

        z = np.random.choice(K, p = probs)
        df.loc[d,"z"] = z

        # Increment counts
        np.add.at(Nk[z], keys, counts)
        Nd[z] += 1

        # Sample new phi and theta
        dir_beta = [np.random.dirichlet(row + beta) for row in Nk]
        phi = np.array([i for i in dir_beta])
        theta = np.array(np.random.dirichlet(Nd + alpha))

# To do:
# Add evaluation metric (that is comparable across different models)
# Also make code a bit safer/general
# Algo converges almost immediately and is then stuck, seems to be a case of multimodality and large influence from taking many phi products


