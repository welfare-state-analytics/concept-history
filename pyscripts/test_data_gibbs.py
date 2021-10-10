from pygibbs import gibbs, gibbs_jit, utils
import json
import numpy as np
import matplotlib.pyplot as plt
from time import time

### Load test data ###
in_path = 'tests/test_data'
out_path = 'test_results'

with open('/'.join([in_path, 'test_stan_data.json'])) as f:
    data = json.load(f)
_,_,_,_,_,_,_,w,doc = list(data.values())
# Index from 0
if min(doc) == 1:
    doc = [x-1 for x in doc]
if min(w) == 1:
    w = [x-1 for x in w]

# Preprocessing
doc, w = utils.rareWords(doc, w, thresh=10)
w, vocab = utils.tokenize(w)
V = len(set(vocab))

# Set parameters
K = 4
alpha = np.ones(K)*0.5
beta = np.ones((K,V))*0.5
epochs = 1000
burn_in = epochs//2
sample_intervals = 5

# Numba JIT version
doc = np.array(doc)
w = np.array(w)

start = time()
theta, phi, Nd, Nk, z, logdensity_jit = gibbs_jit.gibbsSampler(doc, w, V, K, alpha, beta, epochs, burn_in, sample_intervals)
print(f'JIT version finished after {time()-start} sec.')

# Original numpy version
M = len(set(doc))
df = utils.initDataframe(doc, w, M)
df["z"] = np.random.choice(K, M)

start = time()
theta, phi, Nd, Nk, df["z"], logdensity_npy = gibbs.gibbsSampler(df, M, V, K, alpha, beta, epochs, burn_in, sample_intervals)
print(f'NPY version finished after {time()-start} sec.')

# Convergence plot
fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)
fig.suptitle('Gibbs sampling convergence')
ax1.plot(logdensity_jit)
ax2.plot(logdensity_npy)
ax1.title.set_text('Numba JIT version')
ax2.title.set_text('Numpy version ')
plt.show()
plt.close()

# Results on M1 macbook air
# 100 iterations: JIT: 1.67 sec, NPY: 1.9 sec
# 1k iterations: JIT: 1.9 sec, NPY: 18.9 sec
# 10k iterations: JIT: 5 sec, NPY: 189 sec
# 100k iterations: JIT: 39 sec, NPY: 32 min
