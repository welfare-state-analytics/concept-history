from pygibbs import gibbs, utils
import json
import numpy as np
from numba import jit, njit
from progressbar import progressbar
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

w = np.array(w)
doc = np.array(doc)

M, V, = len(set(doc)), len(vocab)

# Set parameters
K = 4
alpha = np.ones(K)*0.5
beta = np.ones((K,V))*0.5
epochs = 10000
burn_in = 200
sample_intervals = 5

def zInit(M, K):
	return np.random.choice(K, M)

@jit(nopython=True)
def gibbsInit(doc, w, z, Nd, Nk, V, K):
	for m in range(M):
		Nd[z[m]] += 1
		idx = np.where(doc == m)[0]

		for i in idx:
			wi = w[i]
			Nk[z[m], wi] += 1

	return (Nd, Nk)

@jit(nopython=True)
def logphi_sampler(logphi, Nk, beta):
    """
    Gibbs sampling step for phi
    """
    for i in range(len(Nk)):
    	logphi[i] = np.log(np.random.dirichlet(Nk[i] + beta[i]))
    return logphi

@jit(nopython=True)
def logtheta_sampler(Nd, alpha):
    """
    Gibbs sampling step for theta
    """
    return np.log(np.random.dirichlet(Nd + alpha))

@jit(nopython=True)
def normalize_logs(x):
    c = x.max()
    lse = c + np.log(np.sum(np.exp(x - c)))
    return np.exp(x - lse)

@jit(nopython=True)
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]

@jit(nopython=True)
def z_sampler(arr, w, z, m, logtheta, logphi, Nd, Nk, K):
	zm = z[m]
	logprobs = logtheta.copy()
	idx = np.where(doc == m)[0]

	# Decrement and compute probs
	Nd[zm] -= 1
	for i in idx:
		wi = w[i]
		Nk[zm, wi] -= 1
		logprobs += logphi[:,wi]

	zm = rand_choice_nb(arr, normalize_logs(logprobs))

	# Increment counts
	Nd[zm] += 1
	for i in idx:
		wi = w[i]
		Nk[zm, wi] += 1

	return (Nd, Nk, zm)

@jit(nopython=True)
def logDensity(logtheta, logphi, Nd, Nk, alpha, beta):
    """
    Joint log density of multinomial clustering model
    """
    return np.multiply((Nd + alpha - 1), logtheta).sum() + \
           np.multiply((Nk + beta - 1), logphi).sum()

def gibbsSampler(doc, w, M, V, K, alpha, beta, epochs, burn_in, sample_intervals):
    """
    Main Gibbs sampling function
    """
    z = zInit(M, K)
    Nd = np.zeros(K, dtype = 'int')
    Nk = np.zeros((K, V), dtype = 'int')
    theta_out = np.zeros(K, dtype = 'float')
    phi_out = np.zeros((K, V), dtype = 'float')
    Nd, Nk = gibbsInit(doc, w, z, Nd, Nk, V, K)

    logphi = np.zeros(np.shape(Nk))
    arr = np.array(range(K))
    logdensity = []
    
    for e in progressbar(range(epochs)):
    	logtheta = logtheta_sampler(Nd, alpha)
    	logphi = logphi_sampler(logphi, Nk, beta)
    	
    	for m in range(M):
	    	Nd, Nk, z[m] = z_sampler(arr, w, z, m, logtheta, logphi, Nd, Nk, K)
    	
    	logdensity.append(logDensity(logtheta, logphi, Nd, Nk, alpha, beta))
    	# Save samples in given intervals
    	if e >= burn_in and e % sample_intervals == 0:
    		phi_out += np.exp(logphi)
    		theta_out += np.exp(logtheta)
    	n_samples = (epochs-burn_in)//sample_intervals
    	
    return (theta_out/n_samples, phi_out/n_samples, Nd, Nk, z, logdensity)

theta, phi, Nd, Nk, z, logdensity = gibbsSampler(doc, w, M, V, K, alpha, beta, epochs, burn_in, sample_intervals)

import matplotlib.pyplot as plt
plt.plot(logdensity)
plt.show()
plt.cla()



