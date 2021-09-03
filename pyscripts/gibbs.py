import numpy as np
import pandas as pd
import json
from collections import Counter
import progressbar
from scipy.special import logsumexp
import matplotlib.pyplot as plt

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

def gibbsInit(df, M, V, K, alpha, beta):
    # Initialize z uniformly at random
    df["z"] = np.random.choice(K, M)

    # Create sufficient statistic matrices
    Nd = np.zeros(K, dtype = 'int')
    Nk = np.zeros((K, V), dtype = 'int')

    # Compute counts
    for doc in range(len(df)):
        d,w,z = df.loc[doc]
        Nd[z] += 1
        np.add.at(Nk[z], w, 1)

    # Initiate theta,phi as regularized sample estimates
    theta = (Nd + alpha) / (Nd + alpha).sum()
    phi = (Nk + beta) / (Nk + beta).sum(axis=1)[:, np.newaxis]

    return(theta, phi, Nd, Nk)

def gibbsStep(w,z,theta,phi,Nd,Nk,alpha,beta):
    # Decrement counts
    np.subtract.at(Nk[z], w, 1)
    Nd[z] -= 1

    # Sample z
    logprobs = np.log(theta) + np.log(np.take(phi, w, axis=1)).sum(axis=1)
    z = np.random.choice(K, p = np.exp(logprobs - logsumexp(logprobs)))

    # Increment counts
    np.add.at(Nk[z], w, 1)
    Nd[z] += 1

    # Sample new phi and theta
    dir_beta = [np.random.dirichlet(row + beta[i]) for i,row in enumerate(Nk)]
    phi = np.array([i for i in dir_beta])
    theta = np.array(np.random.dirichlet(Nd + alpha))

    return (theta, phi, Nd, Nk, z)

def logDensity(theta,phi,Nd,Nk,alpha,beta):
    return np.multiply((Nd + alpha - 1), np.log(theta)).sum() + \
           np.multiply((Nk + beta - 1), np.log(phi)).sum()

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

### Data preprocessing ###
doc, w = rareWords(doc, w, thresh=5)

# Create vocab
vocab = set(w)
vocab = {token:i+1 for i, token in enumerate(vocab)}

# Create df
df = pd.DataFrame(np.array((doc,w)).T)
df.columns = ["doc", "w"]
df = df.groupby('doc')['w'].apply(list).reset_index()

# Hyperparameters, etc.
K = 4
M = len(set(doc))
N = len(w)
V = len(vocab)
alpha = np.array([0.5]*K)
beta = np.array([[0.5]*V]*K)

# Run settings
burn_in = 200
epochs = 1000
sample_intervals = 50

# Output objects
theta_out = np.zeros(K, dtype = 'float')
phi_out = np.zeros((K, V), dtype = 'float')
z_out = np.zeros((len(df),int((epochs-burn_in)/sample_intervals)))

# Initialize and run
theta, phi, Nd, Nk = gibbsInit(df, M, V, K, alpha, beta)
logdensity = []
t = 0

for epoch in progressbar.progressbar(range(epochs)):
    for m in range(M):
        w,z = df.loc[m,["w","z"]]
        theta, phi, Nd, Nk, df.loc[m,"z"] = gibbsStep(w,z,theta,phi,Nd,Nk,alpha,beta)
    logdensity.append(logDensity(theta,phi,Nd,Nk,alpha,beta))

    # Save samples in given intervals
    if epoch >= burn_in and epoch % sample_intervals == 0:
        z_out[:,t] = df["z"]
        t += 1
        phi_out = (t-1)/t * phi_out + 1/t * phi
        theta_out = (t-1)/t * theta_out + 1/t * theta

# Save phi, theta, alogdensity sampled z
np.save(f'{out_path}/phi.npy', phi_out)
np.save(f'{out_path}/theta.npy', theta_out)
np.save(f'{out_path}/z.npy', z_out)
np.save(f'{out_path}/logdensity.npy', np.array(logdensity))

# Save vocab
with open(f"{out_path}/vocab.json", "w") as outfile: 
        json.dump(vocab, outfile, cls=NpEncoder)

plt.plot(logdensity)
plt.savefig(f'{out_path}/logdensity.png')

