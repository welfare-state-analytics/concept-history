import numpy as np
import pandas as pd
from pygibbs import gibbs, utils
import json
import matplotlib.pyplot as plt
from operator import itemgetter

# Load simulated dataset
with open('tests/test_data/test_stan_data.json', 'r') as f:
	d = json.load(f)

M, V, N, K = itemgetter('M', 'V', 'N', 'K')(d)
print(f"documents: {M}, vocabulary size: {V}, word tokens: {N}, topics: {K}")

alpha, beta = itemgetter('alpha', 'beta')(d)
print(f"Alpha has length {np.shape(alpha)} == K")
print(f"Beta has length {np.shape(beta)} == V\n")

doc, w = itemgetter('doc', 'w')(d)
w, vocab = utils.tokenize(w)

# Initialize dataframe
df = utils.initDataframe(doc, w, M)
df["z"] = np.random.choice(K, M)

print('Dataframe used looks like:')
print(df.head())

# Settings
epochs	= 100
burn_in	= 25
sample_intervals = 5

theta, phi, Nd, Nk, df["z"], logdensity, posterior = \
            gibbs.gibbsSampler(df, M, V, K, alpha, beta, epochs, burn_in, sample_intervals)

f = plt.plot(list(range(len(logdensity))), logdensity)
plt.show()
plt.close()