import enum
import numpy as np
import pandas as pd
import os,json
from collections import Counter
from progressbar import progressbar
from itertools import dropwhile
from scipy.special import logsumexp




# Location of test_data
#path = '/home/robinsaberi/Git_Repos/concept-history/tests/test_data'
#files = ['test_stan_data.json','test_theta.json','test_phi.json']
#
# Load files
#lst = []
#for i in range(len(files)):
#    with open(path+'/'+files[i]) as f:
#        lst.append(json.load(f))
#
#data,theta_true,phi_true = lst
#theta_true = np.array((1,2,3,4)) / (1+2+3+4) # Bug in git data, enter manually until fixed
#phi_true = np.array(phi_true)
#K,V,M,N,alpha,beta,z_true,w,doc = list(data.values())
#
#z_true = [x-1 for x in z_true] # Index from 0

# Riksdagen data
path = '/home/robinsaberi/Git_Repos/concept-history/pyscripts/test_context_window_data.json'
with open(path) as f:
    data = json.load(f)# .values()
    w,file,n = data.values()

# Remove rare words
c = Counter(x for xs in w for x in set(xs))
rare_words = set([key for key, value in c.items() if value < 1000000000000])

for lst in w:
    for i, word in enumerate(lst):
        if word in rare_words:
            del lst[i]
    print(lst)
    if len(lst) == 0:
        print(lst)
        print('oops')
        
# Check that no document is completely empty

#K = 10
#alpha,beta = 0.5,0.5
#beta = 0.01
#
## Create df
#df = pd.DataFrame(np.array((doc,w)).T)
#df.columns = ["doc", "w"]
#
#wcount = Counter(w)
#print(len(wcount))
#for key, count in dropwhile(lambda key_count: key_count[1] >= 5, wcount.most_common()):
#    del wcount[key]
#print(len(wcount))
#
#print(2229 / 11678)
#
#df = df.groupby('w').filter(lambda x : (x['w'].count()>=5).any()) # somethings wrong ...
#
#
#df = df.groupby('doc')['w'].apply(list).reset_index()



# Test data
#df["z"] = np.random.choice(K, M) - 1
#df["n"] = [int(N/M)]*M

# Riksdagen data
#df["z"] = np.random.choice(K, M) - 1
#df["n"] = n
#
## Create sufficient statistic matrices
#Nd = np.zeros(K, dtype = 'int')
#Nk = np.zeros((K,V), dtype = 'int')
#
#for doc in range(len(df)):
#    d,w,z,n = df.loc[doc]
#    Nd[z] += 1
#    np.add.at(Nk[z], w, 1)
#
## Initiate theta,phi as unbiased sample estimates
#theta = (Nd + alpha) / (Nd + alpha).sum()
#phi = (Nk + beta) / (Nk + beta).sum(axis=1)[:, np.newaxis]
#
## Run algo
#epochs = 50
#
#for epoch in progressbar(range(epochs)):
#    for doc in range(len(df)):
#        d,w,z,n = df.loc[doc]
#
#        # Get word counts for document
#        w_count = Counter(w)
#        counts = list(w_count.values())
#        keys = list(w_count.keys())
#
#        # Decrement counts
#        np.subtract.at(Nk[z], keys, counts)
#        Nd[z] -= 1
#
#        # Sample sense
#        log_probs = np.log(theta.copy())
#        
#
#        
#        for i in w:
#            log_probs += np.log(phi[:,i])
#        print(doc)
#        print(np.min(phi))
#
#                
#        z = np.random.choice(K, p = np.exp(log_probs - logsumexp(log_probs))) # make them sample from logits
#        
#        df.loc[d,"z"] = z
#
#        # Increment counts
#        np.add.at(Nk[z], keys, counts)
#        Nd[z] += 1
#
#        # Sample new phi and theta
#        dir_beta = [np.random.dirichlet(row + beta) for row in Nk]
#        phi = np.array([i for i in dir_beta])
#        theta = np.array(np.random.dirichlet(Nd + alpha))
#
## To do:
## Add evaluation metric (that is comparable across different models)
## Also make code a bit safer/general
## Algo converges almost immediately and is then stuck, seems to be a case of multimodality and large influence from taking many phi products
#
#
#