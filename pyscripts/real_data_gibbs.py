from pygibbs import gibbs, utils
import os, json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from time import time
import argparse
import re

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    K = config["hyper"]["K"]
    a = config["hyper"]["alpha"]
    b = config["hyper"]["beta"]
    epochs = config["hyper"]["epochs"]
    burn_in = config["hyper"]["burn_in"]
    sample_intervals = config["hyper"]["sample_intervals"]
    files = [f for f in os.listdir(config["paths"]["data"]) if
            f.endswith('.json') and
            f[0 in config["projects"]] and
            int(re.findall(r'\d+', f)[0]) in config["window_sizes"]]
    
    for file in files:
        with open(os.path.join(config["paths"]["data"], file)) as f:
            data = json.load(f)

        doc = data["doc"]
        w = data["w"]
        doc, w = utils.rareWords(doc, w, thresh=10)
        w, vocab = utils.tokenize(w)
        M, V = len(set(doc)), len(set(vocab))
        df = utils.initDataframe(doc, w, M)

        for k in K:
            start = time()
            alpha = np.ones(k)*a
            beta = np.ones((k,V))*b
            df["z"] = np.random.choice(k, M)

            # Run Gibbs sampler
            theta, phi, Nd, Nk, df["z"], logdensity, posterior = \
            gibbs.gibbsSampler(df, M, V, k, alpha, beta, epochs, burn_in, sample_intervals)

            # Store results
            out = '_'.join([file.split('.')[0], 'topic', str(k)])
            out = os.path.join(config["paths"]["results"], out, 'model')
            print(out)
            try:
                os.mkdir(out)
            except:
                pass
            
            np.save(os.path.join(out, 'theta.npy'), theta)
            np.save(os.path.join(out, 'phi.npy'), phi)
            np.save(os.path.join(out, 'Nd.npy'), Nd)
            np.save(os.path.join(out, 'Nk.npy'), Nk)
            np.save(os.path.join(out, 'z.npy'), np.array(df["z"]))
            np.save(os.path.join(out, 'posterior.npy'), posterior)
            np.save(os.path.join(out, 'logdensity.npy'), logdensity)
            with open(os.path.join(out, 'vocab.json'), "w") as f:
                json.dump(vocab, f)

            print(f'Model {out.split("/")[-2]} finished after {(time()-start)//60} minutes.')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--config", type=str)
    args = argparser.parse_args()
    main(args)
