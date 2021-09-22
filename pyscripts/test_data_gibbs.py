from pygibbs import gibbs, utils
import json
import numpy as np
import matplotlib.pyplot as plt

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
M, V = len(set(doc)), len(vocab)

# Set parameters
K = 4
alpha = np.ones(K)*0.5
beta = np.ones((K,V))*0.5
epochs = 20
burn_in = 5
sample_intervals = 5

# Format data and init z
df = utils.initDataframe(doc, w, M)
df["z"] = np.random.choice(K, M)

# Run algorithm
theta, phi, Nd, Nk, z, logdensity = gibbs.gibbsSampler(df, M, V, K, alpha, beta, epochs, burn_in, sample_intervals)

# Postprocessing
results = utils.distinctWords(Nk, beta, vocab, n=6)

plt.plot(logdensity)
plt.show()
plt.cla()
