import numpy as np
from scipy.special import logsumexp
from progressbar import progressbar

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

def gibbsSampler(df, M, V, K, alpha, beta, epochs, burn_in, sample_intervals, sample_post=True):
    """
    Main Gibbs sampling function
    """
    posterior = False
    if sample_post == True:
        posterior = np.zeros((M, K), dtype=int)
        row_indices = list(range(M))

    Nd, Nk = gibbsInit(df, V, K)
    theta_out = np.zeros(K, dtype = 'float')
    phi_out = np.zeros((K, V), dtype = 'float')
    n_samples = 0
    logdensity = []
    for e in progressbar(range(epochs)):
        logtheta = logtheta_sampler(Nd, alpha)
        logphi = logphi_sampler(Nk, beta)
        Nd, Nk, df["z"] = z_sampler(df,logtheta,logphi,Nd,Nk,M,K)
        logdensity.append(logDensity(logtheta,logphi,Nd,Nk,alpha,beta))

        # Save samples in given intervals
        if e >= burn_in and (e+1) % sample_intervals == 0:
            theta_out += np.exp(logtheta)
            phi_out += np.exp(logphi)
            n_samples += 1
            if sample_post == True:
                np.add.at(posterior, tuple([row_indices, df["z"]]), 1)

    return (theta_out/n_samples, phi_out/n_samples, Nd, Nk, df["z"], logdensity, posterior)
