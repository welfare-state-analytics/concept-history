"""
Python module for multinomial cluster model inference for word sense induction using Gibbs sampling.
The module utilizes Numba just-in-time compilation and has been tested to work on the M1-macbook.
"""
from numba.np.ufunc import parallel
import numpy as np
from progressbar import progressbar
from numba import njit, jit, prange


@jit(nopython=True)
def gibbsInit(doc, w, z, M, V, K):
	"""
	Creates the sufficient statistic matrices that the algorithm operates on.
	"""
	Nd = np.zeros(K, dtype='int')
	Nk = np.zeros((K, V), dtype='int')
	# For doc
	for m in range(M):
		Nd[z[m]] += 1
		idx = np.where(doc == m)[0]
		# For word token in doc
		for i in idx:
			Nk[z[m], w[i]] += 1

	return (Nd, Nk)


@jit(nopython=True)
def gibbsInit_para(doc, w, z, M, V, K):
	"""
	Creates the sufficient statistic matrices that the algorithm operates on.
	"""
	Nd = np.zeros(K, dtype='int')
	Nk = np.zeros((K, V), dtype='int')
	# For doc
	for m in range(M):
		Nd[z[m]] += 1
		idx = np.where(doc == m)[0]
		# For word token in doc
		for i in idx:
			Nk[z[m], w[i]] += 1

	return (Nd, Nk)


@jit(nopython=True)
def logphi_sampler(logphi, Nk, K, beta):
    """
    Gibbs sampling step for phi.
    """
    for k in range(K):
    	logphi[k] = np.log(np.random.dirichlet(Nk[k] + beta[k]))


@jit(nopython=True)
def logtheta_sampler(logtheta, Nd, alpha):
    """
    Gibbs sampling step for theta.
    """
    logtheta = np.log(np.random.dirichlet(Nd + alpha))


@jit(nopython=True)
def normalize_logs(x):
	"""
	Creates probability distribution from unnormalized log probabilites.
	"""
	c = x.max()
	lse = c + np.log(np.sum(np.exp(x - c)))
	return np.exp(x - lse)


@jit(nopython=True)
def rand_choice_nb(arr, prob):
    """
    Numba does is unable to compile np.random.choice, so a work-around is used.
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


@jit(nopython=True)
def z_sample_doc(arr, doc, w, z, m, logtheta, logphi, Nd, Nk, K):
	"""
	Samples a sense and updates counts for a single document.
	"""
	zm = z[m]
	logprobs = logtheta.copy()
	idx = np.where(doc == m)[0]

	# Decrement counts and compute probs
	Nd[zm] -= 1
	for i in idx:
		wi = w[i]
		Nk[zm, wi] -= 1
		logprobs += logphi[:, wi]

	# Sample sense
	zm = rand_choice_nb(arr, normalize_logs(logprobs))
	z[m] = zm

	# Increment counts
	Nd[zm] += 1
	for i in idx:
		wi = w[i]
		Nk[zm, wi] += 1


@jit(nopython=True)
def z_sample_doc_para(arr, doc, w, m, logtheta, logphi, K):
	"""
	Samples a sense and updates counts for a single document.
	"""
	logprobs = np.zeros(K)
	logprobs += logtheta

	idx = np.where(doc == m)[0]

	# Compute probs
	for i in range(len(idx)):
		logprobs += logphi[:, w[idx[i]]]

	# Sample sense
	return rand_choice_nb(arr, normalize_logs(logprobs))


@jit(nopython=True)
def z_sampler(arr, doc, w, z, logtheta, logphi, Nd, Nk, M, K):
	"""
	Performs a single pass through the dataset sampling sense and updating counts for each document.
	"""
	for m in range(M):
		z_sample_doc(arr, doc, w, z, m, logtheta, logphi, Nd, Nk, K)


@njit(parallel=True)
def z_sampler_para(arr, doc, w, z, logtheta, logphi, M, K):
	"""
	Performs a single pass through the dataset sampling sense and updating counts for each document.
	"""
	for m in prange(M):
		logprobs = np.zeros(K)
		logprobs += logtheta

		idx = np.where(doc == m)[0]

		# Compute probs
		for i in prange(len(idx)):
			logprobs += logphi[:, w[idx[i]]]
		
		z[m] = rand_choice_nb(arr, normalize_logs(logprobs))


@jit(nopython=True)
def gibbsIteration(arr, doc, w, z, logtheta, logphi, Nd, Nk, M, K, alpha, beta):
	"""
	Performs one iteration of Gibbs sampling over the whole dataset.
	"""
	logtheta_sampler(logtheta, Nd, alpha)
	logphi_sampler(logphi, Nk, K, beta)
	z_sampler(arr, doc, w, z, logtheta, logphi, Nd, Nk, M, K)


@jit(nopython=True)
def gibbsIteration_para(arr, doc, w, z, logtheta, logphi, Nd, Nk, M, V, K, alpha, beta):
	"""
	Performs one iteration of Gibbs sampling over the whole dataset.
	"""
	logtheta_sampler(logtheta, Nd, alpha)
	logphi_sampler(logphi, Nk, K, beta)
	z_sampler_para(arr, doc, w, z, logtheta, logphi, M, K)
	Nd, Nk = gibbsInit_para(doc, w, z, M, V, K)
	return (Nd, Nk)


@jit(nopython=True)
def logDensity(logtheta, logphi, Nd, Nk, alpha, beta):
    """
    Joint log density for multinomial clustering model.
    """
    return np.multiply((Nd + alpha - 1), logtheta).sum() + \
           np.multiply((Nk + beta - 1), logphi).sum()


def gibbsSampler(doc, w, V, K, alpha, beta, epochs, burn_in, sample_intervals):
	# Inputs
	M = len(set(doc))
	z = np.random.choice(K, M)
	Nd, Nk = gibbsInit(doc, w, z, M, V, K)
	logtheta = np.zeros(K)
	logphi = np.zeros((K, V))
	arr = np.array(range(K))
	# Outputs
	n_samples = (epochs-burn_in)//sample_intervals
	theta_out = logtheta.copy()
	phi_out = logphi.copy()
	logdensity = []
	for e in progressbar(range(epochs)):
		Nd, Nk = gibbsIteration_para(arr, doc, w, z, logtheta, logphi, Nd, Nk, M, V, K, alpha, beta)
		logdensity.append(logDensity(logtheta, logphi, Nd, Nk, alpha, beta))
	# Save samples in given intervals
	if e >= burn_in and e % sample_intervals == 0:
		phi_out += np.exp(logphi)
		theta_out += np.exp(logtheta)
	return(theta_out/n_samples, phi_out/n_samples, Nd, Nk, z, logdensity)

# gibbsIteration(arr, doc, w, z, logtheta, logphi, Nd, Nk, M, K, alpha, beta)
#import json
#import utils
#
#### Load test data ###
#in_path = 'data/context_windows'
#file = 'f_window_5.json'
#
#with open('/'.join([in_path, file])) as f:
#    data = json.load(f)
#doc = data["doc"]
#w = data["w"]
#
#doc, w = utils.rareWords(doc, w, thresh=10)
#w, vocab = utils.tokenize(w)
#V = len(set(vocab))
#
## Set parameters
#K = 5
#alpha = np.ones(K)*0.5
#beta = np.ones((K,V))*0.5
#
#M = len(set(doc))
#z = np.random.choice(K, M)
#
#Nd, Nk = gibbsInit(doc, w, z, M, V, K)
#logtheta = np.zeros(K)
#logphi = np.zeros((K, V))
#logtheta_sampler(logtheta, Nd, alpha)
#logphi_sampler(logphi, Nk, K, beta)
#
#
#arr = np.array(range(K))
#
#z_sampler_para(arr, doc, w, z, logtheta, logphi, M)
#z_sampler_para.parallel_diagnostics(level=4)