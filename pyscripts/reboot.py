"""
Runs existing model for longer in order to sample from the posterior.
"""
from distutils import util
from pygibbs import gibbs, utils
import os, json
import numpy as np
import subprocess

def preprocessing(in_path, file):
    with open('/'.join([in_path, file])) as f:
        data = json.load(f)
    doc = data["doc"]
    w = data["w"]

    # Index from 0
    if min(doc) == 1:
        doc = [x-1 for x in doc]
    if min(w) == 1:
        w = [x-1 for x in w]

    # Preprocessing
    doc, w = utils.rareWords(doc, w, thresh=10)
    w, vocab = utils.tokenize(w)
    M = len(set(doc))
    V = len(set(vocab))

    return (doc, w, vocab, M, V)

data_path = 'data/context_windows'

# for folder
# Load model
folder = 'f_window_5_topic_5'
results_path = os.path.join('results/first-run', folder)

# Load data
file = '_'.join(folder.split('_')[:3]) + '.json'
doc, w, vocab, M, V = preprocessing(data_path, file)
df = utils.initDataframe(doc, w, M)

# Load model
Nd = np.load(os.path.join(results_path, 'Nd.npy'))
Nk = np.load(os.path.join(results_path, 'Nk.npy'))
theta = np.load(os.path.join(results_path, 'theta.npy'))
phi = np.load(os.path.join(results_path, 'phi.npy'))
df["z"] = np.load(os.path.join(results_path, 'z.npy'))

K = len(set(df["z"]))
alpha = np.ones(K)*0.5
beta = np.ones((K, V))*0.5

epochs = 1000
burn_in = 0
sample_intervals = 5

theta, phi, Nd, Nk, z, logdensity, posterior = gibbs.gibbsSampler(df, M, V, K, alpha, beta, epochs, burn_in, sample_intervals)

# Save and push results
posterior_path = os.path.join(results_path, 'posterior.npy')
np.save(posterior_path, posterior)
utils.git_auto_push(posterior_path, 'added posterior from rebooted model')
