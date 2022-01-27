"""
TODO:
- optimize _get_state, very inefficient
- running c=20 becomes infeasible, takes 3 days
"""
import os, json
from pygibbs import utils
import numpy as np
import pandas as pd
import subprocess as sp
import gzip
import re
import progressbar
import time

def cleanYear(year):
    if len(year) == 4:
        return int(year)
    return int(year[:4])

def mallet_dataset(mallet_path, input, results_path, keep_sequence=True):
	"""
	Creates a (training) dataset in .mallet format.
	"""
	cmd = [ mallet_path,
		"import-dir",
		"--input", input,
		"--output", os.path.join(results_path, 'train.mallet'),
		"--keep-sequence", str(keep_sequence),
		]
	proc = sp.Popen(cmd)
	proc.wait()
	print(f'Dataset stored in {results_path}.')

def _logperplex(mallet_output, print_out=True):
	"""
	Scrapes log perplexity from mallet console output.
	"""
	lp = []
	for line in str(mallet_output).split('\\n'):
		if print_out==True: 
			print(line.replace('\\t', ' '))
		if "LL/token" in line:
			lp.append(float(line.split()[-1].replace(',', '.')))
	return np.array(lp)

def _get_state(results_path):
	"""
	Reads mallet binary state file and creates a formatted dataset from it.
	"""
	start = time.time()
	variables = ["doc", "w", "z", "word", "pos", "file", "target"]
	data = []
	f = gzip.open(os.path.join(results_path, "state.gz"), "rb")
	
	for i, line in enumerate(f):
		if i >= 3:
			s = str(line).split()
			file = s[1].split('/')[-1]
			target = file.split('_')[2]
			doc = int(s[0].split("b'")[-1])			
			pos = int(s[2])
			w = int(s[3])
			word = s[4]
			z = int(s[5].split('\\n')[0])
			data.append([doc, w, z, word, pos, file, target])
	f.close()
	df = pd.DataFrame(data, columns=variables)
	print(f'State retrieved after {time.time()-start} seconds.')
	return df

def _get_results(results_path):
	"""
	Generates results (numpy objects and vocab) from state
	TODO: Would be nice to merge with previous function and skip the double loop
	"""
	df = _get_state(results_path)
	start = time.time()

	vocab = {}
	M = max(df["doc"])+1
	V = len(set(df["w"]))
	K = len(set(df["z"]))
	Nd = np.zeros((M, K), dtype=int)
	Nk = np.zeros((K, V), dtype=int)
	Z = np.zeros(M, dtype=int)
	#years = np.zeros(M, dtype=int)
	years = []
	#targets = np.zeros(M, dtype=str)
	targets = []

	bar = progressbar.ProgressBar(max_value=len(df))
	for i, row in df.iterrows():
		m, w, z, word, pos, file, target = row[["doc", "w", "z", "word", "pos", "file", "target"]]

		if word not in vocab:
			vocab[word] = w

		# Switch to np.add.at(coordinates)
		Nd[m,z] += 1
		Nk[z,w] += 1
		
		# This part is broken...
		# Switch to:
		# 1. Change words from \xcc --> ä
		# 2. Potentially switch to if word in list(set(targets))
		# 3. Optionally wrap the gensim wrapper...
		if word == target:
			Z[m] = z
			year = cleanYear(re.findall(r'prot-([0-9]+)-', file)[0])
			#years[m] = year
			years.append(year)
			#targets[m] = target
			targets.append(target)
			
		bar.update(i)

	theta = Nd / Nd.sum(axis=1)[:,np.newaxis]
	phi = Nk / Nk.sum(axis=1)[:,np.newaxis]

	results = [theta, phi, Nd, Nk, Z, vocab, years, targets]
	print(f'Results retrieved after {time.time()-start} seconds.')
	return results

def mallet_lda(mallet_path, results_path, K, alpha, beta, epochs, print_out=True):
	"""
	Main LDA function. Runs mallet train-topics and gathers results from final state.
	Returns: [theta, phi, Nd, Nk, Z, vocab, logperplexity]
	TODO: Refactor, this function does too much
	"""
	cmd = [ mallet_path,
			"train-topics",
			"--input", os.path.join(results_path, 'train.mallet'),
			"--output-model", os.path.join(results_path, "lda.model"),
			"--output-state", os.path.join(results_path, "state.gz"),
			"--num-topics", str(K),
			"--alpha", str(alpha*K),
			"--beta", str(beta),
			"--num-iterations", str(epochs),
			]

	start = time.time()
	proc = sp.Popen(cmd, stderr = sp.PIPE)
	_, mallet_output = proc.communicate()
	print(f'FINISHED after {time.time()-start}')	
	lp = _logperplex(mallet_output, print_out=print_out)

	# Works all the way here
	return lp

# Run
mallet_path = os.path.expanduser('~/mallet-2.0.8/bin/mallet')
K, alpha, beta = 5, 0.5, 0.5
epochs = 10000

# for dataset in
K = [5, 10]
projects = ['f', 'j']
window_sizes = [5, 10, 20]

for c in window_sizes:
	for p in projects:
		for k in K:
			data_dir = f'data/mallet_windows/{p}_window_{str(c)}'
			#data_dir = os.path.expanduser('~/mallet-2.0.8/sample-data/web/en') # TEST FOR DEBUGGING

			# Create output folder
			subdirs = ['results/lda-models', p, f'window_{c}_topic_{k}', 'model']
			for i in range(len(subdirs)):
				out = '/'.join(subdirs[:i+1])
				exec('try:os.mkdir(out)\nexcept:pass')

			# Main LDA functions
			#mallet_dataset(mallet_path, data_dir, out)
			#logperplexity = mallet_lda(
				#mallet_path, out, K=k, alpha=alpha, beta=beta, epochs=epochs)

			theta, phi, Nd, Nk, Z, vocab, years, targets = _get_results(out)
			print(len(years))
			print(len(targets))
			print(np.shape(Nd))
			from collections import Counter
			print(Counter(years))
			print(Counter(targets))

			# Store results
			np.save(os.path.join(out, 'theta.npy'), theta)
			np.save(os.path.join(out, 'phi.npy'), phi)
			np.save(os.path.join(out, 'Nd.npy'), Nd)
			np.save(os.path.join(out, 'Nk.npy'), Nk)
			np.save(os.path.join(out, 'z.npy'), Z)
			np.save(os.path.join(out, 'time.npy'), years)
			np.save(os.path.join(out, 'target.npy'), targets)
			#np.save(os.path.join(out, 'logperplexity.npy'), logperplexity)
			with open(os.path.join(out, 'vocab.json'), 'w') as f:
				json.dump(vocab, f, ensure_ascii=True, indent=4)

