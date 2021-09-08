import numpy as np
import pandas as pd
import json
from collections import Counter
import progressbar
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import time
from bidict import bidict

# Create a package...

# Make output vocab json compatible
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)

# Removes rare words from documents
def rareWords(doc, w, thresh=5):
    c = Counter(w)
    rare_words = set([key for key, value in c.items() if value < thresh])

    idx= []
    for i in range(len(w)):
        if w[i] not in rare_words:
            idx.append(i)

    w = [w[i] for i in idx]
    doc = [doc[i] for i in idx]

    return(doc, w)

# Create vocab
def tokenize(w):
    vocab = set(w)
    vocab = {token:i for i, token in enumerate(vocab)}
    w = [vocab[token] for token in w]
    return w, vocab

def initDataframe(doc,w, M):
    df = pd.DataFrame(np.array((doc,w)).T)
    df.columns = ["doc", "w"]
    df = df.groupby('doc')['w'].apply(list).reset_index()
    df["z"] = np.random.choice(K, M)
    return df

def gibbsInit(df, V, K):

    # Create sufficient statistic matrices
    Nd = np.zeros(K, dtype = 'int')
    Nk = np.zeros((K, V), dtype = 'int')

    # Compute counts
    for doc in range(len(df)):
        w,z = df.loc[doc,["w","z"]]
        Nd[z] += 1
        np.add.at(Nk[z], w, 1)

    return(Nd, Nk)

# Pretty inefficient...
def logphi_sampler(Nk, beta):
    dir_beta = [np.random.dirichlet(row + beta[i]) for i,row in enumerate(Nk)]
    phi = np.array(dir_beta)
    return np.log(phi)

def logtheta_sampler(Nd, alpha):
    theta = np.array(np.random.dirichlet(Nd + alpha))
    return np.log(theta)

def z_sampler(w,z,logtheta,logphi,Nd,Nk):
    # Decrement counts
    np.subtract.at(Nk[z], w, 1)
    Nd[z] -= 1

    # Sample z
    logprobs = logtheta + np.take(logphi, w, axis=1).sum(axis=1)
    z = np.random.choice(K, p = np.exp(logprobs - logsumexp(logprobs)))

    # Increment counts
    np.add.at(Nk[z], w, 1)
    Nd[z] += 1

    return (Nd, Nk, z)

# Evaluation function
def logDensity(logtheta,logphi,Nd,Nk,alpha,beta):
    return np.multiply((Nd + alpha - 1), logtheta).sum() + \
           np.multiply((Nk + beta - 1), logphi).sum()

# Returns top n p(z=k|w)
def distinctWords(Nk, beta, vocab, n=10):
    vocab_rev = bidict(vocab).inv

    K = len(Nk)
    word_list = []
    results = (Nk+beta) / (Nk+beta).sum(axis=0)[np.newaxis]

    for k in range(K):
        indices = (-results[k]).argsort()[:n]
        word_list.append([vocab_rev.get(key) for key in indices])

    return np.array(word_list).T

def gibbsSampler(df, M, V, K, alpha, beta, epochs, burn_in, sample_intervals):

    Nd, Nk = gibbsInit(df, V, K)

    # Output objects
    theta_out = np.zeros(K, dtype = 'float')
    phi_out = np.zeros((K, V), dtype = 'float')
    Nd_out = np.zeros(K, dtype = 'int')
    Nk_out = np.zeros((K, V), dtype = 'int')
    z_out = np.zeros((len(df),int((epochs-burn_in)/sample_intervals)))

    logdensity = []
    t = 0

    for e in progressbar.progressbar(range(epochs)):
        logtheta = logtheta_sampler(Nd, alpha)
        logphi = logphi_sampler(Nk, beta)

        for m in range(M):
            w,z = df.loc[m,["w","z"]]
            Nd, Nk, df.loc[m,"z"] = z_sampler(w,z,logtheta,logphi,Nd,Nk)

        logdensity.append(logDensity(logtheta,logphi,Nd,Nk,alpha,beta))

        # Save samples in given intervals
        if e >= burn_in and e % sample_intervals == 0:
            z_out[:,t] = df["z"]
            t += 1
            phi_out = (t-1)/t * phi_out + 1/t * np.exp(logphi)
            theta_out = (t-1)/t * theta_out + 1/t * np.exp(logtheta)
            Nd_out += Nd
            Nk_out += Nk

    top_words = pd.DataFrame(distinctWords(Nk, beta, vocab, n=6))
    top_words.columns = list(range(K))

    return (theta_out, phi_out, Nd_out/t, Nk_out/t, z_out, logdensity, top_words)

### Load test data ###
#in_path = 'tests/test_data'
#out_path = 'test_results'
#
#with open('/'.join([in_path, 'test_stan_data.json'])) as f:
#    data = json.load(f)
#_,_,_,_,_,_,_,w,doc = list(data.values())
## Index from 0
#if min(doc) == 1:
#    doc = [x-1 for x in doc]
#if min(w) == 1:
#    w = [x-1 for x in w]

### Load corpus data ###
window_sizes = [5, 10, 50, 100]
n_topics = [5, 10]
in_path = f'ctx_window_data'

# Run settings
burn_in = 0
epochs = 10
sample_intervals = 1

projects = ['f','j']
for project in projects:
    for K in n_topics:
        for window_size in window_sizes:
            # Set output path
            out_path = f'results/{project}/window_{window_size}_{K}'
    
            # Load data
            with open('/'.join([in_path, f'ctx_window_{window_size}_{project}.json'])) as f:
                w,doc = json.load(f).values()
    
            ### Data preprocessing ###
            doc, w = rareWords(doc, w, thresh=10)
    
            # Testing
            doc = doc[:10000]
            w = w[:10000]
    
            # Initialize variables
            w, vocab = tokenize(w)
            M, V = len(set(doc)), len(vocab)
            df = initDataframe(doc,w,M)
            alpha = np.array([0.5]*K)
            beta = np.array([[0.5]*V]*K)
    
            # Run algorithm
            theta, phi, Nd, Nk, z, logdensity, top_words = gibbsSampler(df, M, V, K, alpha, beta, epochs, burn_in, sample_intervals)
    
            # Store output objects
            top_words.to_csv(f'{out_path}/top_words.csv', index=False)
    
            np.save(f'{out_path}/theta.npy', theta)
            np.save(f'{out_path}/phi.npy', phi)
            np.save(f'{out_path}/Nd.npy', Nd)
            np.save(f'{out_path}/Nk.npy', Nk)
            np.save(f'{out_path}/z.npy', z)
            np.save(f'{out_path}/logdensity.npy', np.array(logdensity))
    
            with open(f"{out_path}/vocab.json", "w") as outfile: 
                    json.dump(vocab, outfile, cls=NpEncoder)
    
            plt.plot(logdensity)
            plt.savefig(f'{out_path}/logdensity.png')
            plt.cla()