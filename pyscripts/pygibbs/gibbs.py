import numpy as np
import progressbar
from scipy.special import logsumexp
import matplotlib.pyplot as plt

def gibbsInit(df, V, K):
    """
    Creates sufficient statistic matrices Nd and Nk
    """
    Nd = np.zeros(K, dtype = 'int')
    Nk = np.zeros((K, V), dtype = 'int')

    # Compute counts
    for doc in range(len(df)):
        w,z = df.loc[doc,["w","z"]]
        Nd[z] += 1
        np.add.at(Nk[z], w, 1)

    return(Nd, Nk)

def logphi_sampler(Nk, beta):
    """
    Gibbs sampling step for phi
    """
    phi = np.array([np.random.dirichlet(row + beta[i]) for i,row in enumerate(Nk)])
    return np.log(phi)

def logtheta_sampler(Nd, alpha):
    """
    Gibbs sampling step for theta
    """
    theta = np.array(np.random.dirichlet(Nd + alpha))
    return np.log(theta)

def z_sample_step(w, z, logtheta, logphi, Nd, Nk, K):
    """
    Samples 1 topic indicator and updates sufficient statistic counts
    """
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

def z_sampler(df, logtheta, logphi, Nd, Nk, M, K):
    """
    Gibbs sampling step for topic indicator
    """
    Z = np.empty(M, dtype=int)
    for m in range(M):
        w,z = df.loc[m,["w","z"]]
        Nd, Nk, Z[m] = z_sample_step(w,z,logtheta,logphi,Nd,Nk,K)
    return (Nd, Nk, Z)

def logDensity(logtheta, logphi, Nd, Nk, alpha, beta):
    """
    Joint log density of multinomial clustering model
    """
    return np.multiply((Nd + alpha - 1), logtheta).sum() + \
           np.multiply((Nk + beta - 1), logphi).sum()

def gibbsSampler(df, M, V, K, alpha, beta, epochs, burn_in, sample_intervals):

    """
    Main Gibbs sampling function
    """

    Nd, Nk = gibbsInit(df, V, K)

    # Output objects
    theta_out = np.zeros(K, dtype = 'float')
    phi_out = np.zeros((K, V), dtype = 'float')

    logdensity = []
    t = 0

    for e in progressbar.progressbar(range(epochs)):
        logtheta = logtheta_sampler(Nd, alpha)
        logphi = logphi_sampler(Nk, beta)

        Nd, Nk, df["z"] = z_sampler(df,logtheta,logphi,Nd,Nk,M,K)

        logdensity.append(logDensity(logtheta,logphi,Nd,Nk,alpha,beta))

        # Save samples in given intervals
        if e >= burn_in and e % sample_intervals == 0:
            phi_out += np.exp(logphi)
            theta_out += np.exp(logtheta)

    n_samples = (epochs-burn_in)//sample_intervals

    return (theta_out/n_samples, phi_out/n_samples, Nd, Nk, df["z"], logdensity)

## Load test data ###
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
#
#### Load corpus data ###
#window_sizes = [5, 10, 50, 100]
#n_topics = [5, 10]
#in_path = f'ctx_window_data'
#
## Run settings
#burn_in = 0
#epochs = 10
#sample_intervals = 1
#
#projects = ['f','j']
#for project in projects:
#    for K in n_topics:
#        for window_size in window_sizes:
#            # Set output path
#            out_path = f'results/{project}/window_{window_size}_{K}'
#    
#            # Load data
#            with open('/'.join([in_path, f'ctx_window_{window_size}_{project}.json'])) as f:
#                w,doc = json.load(f).values()

# Preprocessing
#doc, w = rareWords(doc, w, thresh=10)
#w, vocab = tokenize(w)
#M, V = len(set(doc)), len(vocab)
#
## Set parameters
#K = 4
#alpha = np.ones(K)*0.5
#beta = np.ones((K,V))*0.5
#epochs = 20
#burn_in = 5
#sample_intervals = 5
#
## Format data and init z
#df = initDataframe(doc, w, M)
#df["z"] = np.random.choice(K, M)
#
## Run algorithm
#theta, phi, Nd, Nk, z, logdensity = gibbsSampler(df, M, V, K, alpha, beta, epochs, burn_in, sample_intervals)
#
## Postprocessing
#results = distinctWords(Nk, beta, vocab, n=6)
#print(results)

# Store output objects
#top_words.to_csv(f'{out_path}/top_words.csv', index=False)
#
#np.save(f'{out_path}/theta.npy', theta)
#np.save(f'{out_path}/phi.npy', phi)
#np.save(f'{out_path}/Nd.npy', Nd)
#np.save(f'{out_path}/Nk.npy', Nk)
#np.save(f'{out_path}/z.npy', z)
#np.save(f'{out_path}/logdensity.npy', np.array(logdensity))
#with open(f"{out_path}/vocab.json", "w") as outfile: 
#        json.dump(vocab, outfile)

#plt.plot(logdensity)
#plt.show()
#plt.savefig(f'{out_path}/logdensity.png')
#plt.cla()




