from pygibbs import gibbs, utils
import os, json
import numpy as np
import matplotlib.pyplot as plt
from time import time

def main(hyperparameters, in_path, out_path):
    K = hyper["K"]
    a = hyper["alpha"]
    b = hyper["beta"]
    epochs = hyper["epochs"]
    burn_in = hyper["burn_in"]
    sample_intervals = hyper["sample_intervals"]
    
    with open(in_path) as f:
        data = json.load(f)
    doc = data["doc"]
    w = data["w"]
    doc, w = utils.rareWords(doc, w, thresh=10)
    w, vocab = utils.tokenize(w)
    M, V = len(set(doc)), len(set(vocab))
    df = utils.initDataframe(doc, w, M)

    for k in K:
        alpha = np.ones(k)*a
        beta = np.ones((k,V))*b
        df["z"] = np.random.choice(k, M)

        # Run Gibbs sampler
        theta, phi, Nd, Nk, df["z"], logdensity, logposterior = \
        gibbs.gibbsSampler(df, M, V, k, alpha, beta, epochs, burn_in, sample_intervals)

        # Store results
        out = '_'.join([out_path, 'topic', str(k)])
        np.save(os.path.join(out, 'theta.npy'), theta)
        np.save(os.path.join(out, 'phi.npy'), phi)
        np.save(os.path.join(out, 'Nd.npy'), Nd)
        np.save(os.path.join(out, 'Nk.npy'), Nk)
        np.save(os.path.join(out, 'z.npy'), np.array(df["z"]))
        np.save(os.path.join(out, 'logposterior.npy'), logposterior)
        np.save(os.path.join(out, 'logdensity.npy'), logdensity)
        with open(os.path.join(out, 'vocab.json'), "w") as f:
            json.dump(vocab, f)

        print(f'{out.split("/")[-1]} finished.')

if __name__ == "__main__":
    # Specify paths and data files
    in_path = 'data/context_windows'
    out_path = 'results/second-run'
    files = [file for file in os.listdir(in_path) if file.endswith('.json')]

    # Take subset for testing
    files = [f for f in files if not any(s in f for s in ['10','20','50','100'])]

    # Set parameters
    hyper = {
    "K": [5, 10],
    "alpha": 0.5,
    "beta": 0.5,
    "epochs": 2000,
    "burn_in": 1000,
    "sample_intervals": 10
    }
    
    for file in files:
        main(
            hyperparameters=hyper,
            in_path = os.path.join(in_path, file),
            out_path = os.path.join(out_path, file.split('.')[0])
            )

