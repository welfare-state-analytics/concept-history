import numpy as np
import pandas as pd
import os, json
from pygibbs import utils
import matplotlib.pyplot as plt
import time
from collections import Counter
import gzip
import sklearn.preprocessing
import progressbar

def get_senses():
	z = np.zeros(M, dtype=int)
	for m in progressbar.progressbar(range(M)):
		z[m] = state.loc[state["#doc"] == m].iloc[data.loc[m,"pos"]]["topic"]
	return z

def rare_words():
	"""
	TODO: Add args and move
	"""
	thresh = 5
	word_counts = Counter(' '.join(data["w"]).split())
	for m, row in data.iterrows():
		words = []
		for i, word in enumerate(row["w"].split()):
			if word_counts.get(word) > thresh or word in target_words:
				words.append(word)
			elif i < row["pos"]:
				data.loc[m, "pos"] -= 1
		data.loc[m,"w"] = ' '.join(words)
		if len(words) == 0:
			print(f'Document {i} is empty after filtering rare words and is removed.')
			data = data.drop(i)
	data.reset_index(drop=True)
	data["doc"] = data.index
	data.to_csv(os.path.join(path_model, 'data.csv'), index=False)

def extract_params(state):
	"""
	Extracts model parameters and count matrices from state df.
	"""
	M, K = max(state["#doc"])+1, len(set(state["topic"]))
	Nd = np.zeros((M, K))
	Nk = np.zeros((K, V))
	np.add.at(Nd, tuple((state["#doc"], state["topic"])), 1)
	np.add.at(Nk, tuple((state["topic"], state["typeindex"])), 1)
	theta = Nd / Nd.sum(axis=1)[:,np.newaxis]
	phi = Nk / Nk.sum(axis=1)[:,np.newaxis]
	return (theta, phi, Nd, Nk)

def add_meta_data_to_state(state, meta):
	variables = ["year", "gender", "party_abbrev", "lemmas"]
	doclenghts = Counter(state["#doc"])
	data = []
	for m in progressbar.progressbar(range(max(state["#doc"])+1)):
		data.extend([meta.loc[m, variables].tolist()] * doclenghts[m])
	data = pd.DataFrame(data, columns=variables)
	state[variables] = data

with open('data/stopwords-sv.txt','r') as f:
	stopwords = f.read().splitlines()
periods = np.array([list(np.arange(1920+(10*i), 1920+(10*(i+1)))) for i in range(11)])
colors = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1',\
		  '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
alpha, beta = 0.5, 0.5
#p, c, k = 'f', 5, 5

# Load data
for p in ['f']:
	with open('data/target-words.json', 'r') as f:
		target_words = json.load(f)["target_" + p]
	for c in [20]:
		for k in [20]:
			if c == 5 and (k==20 or k==50):
				continue

			start = time.time()

			path_project = f'results/lda-models'
			path_model = f'results/lda-models/{p}/window_{c}_topic_{k}/model'
			path_analysis = f'results/lda-models/{p}/window_{c}_topic_{k}/analysis'
			path_meta = f'results/lda-models/{p}/meta.csv'
			path_state = os.path.join(path_model, 'state.gz')

			color = colors if k <= 10 else None

			# Load metadata
			meta = pd.read_csv(path_meta)
			meta["z"] = np.load(os.path.join(path_model, 'z.npy'))
			print(f'Step 1 finished after {time.time()-start}')

			# Load state and add metadata
			state = pd.read_csv(path_state, compression='gzip', sep=' ', skiprows=[1,2])
			add_meta_data_to_state(state, meta)
			state["z"] = state["topic"]
			print(f'Step 2 finished after {time.time()-start}')

			# Load model
			lp = np.load(f'{path_model}/logperplexity.npy')
			vocab = dict(zip(state.type, state.typeindex))
			M, V, K = len(meta), len(vocab), len(set(state["topic"]))
			theta, phi, Nd, Nk = extract_params(state)
			print(f'Step 3 finished after {time.time()-start}')

			# Sort z
			meta["z"], theta, phi, Nd, Nk, z_sorted = utils.z_sorter(meta["z"], theta, phi, Nd, Nk)
			
			# STUCK HERE
			print(state["z"])
			state = state.replace({'z':{key:value for key, value in zip(list(range(K)), z_sorted)}})
			print(state["z"])
			print(f'Step 5 finished after {time.time()-start}')

			#meta.to_csv(f'{path_model}/meta.csv', index=False)
			#state.to_csv(f'{path_model}/state.csv', index=False)

			lemmas = set(meta["lemmas"])
			genders = ["woman", "man"]
			parties = [p for p in list(set(meta["party_abbrev"])) if str(p) != 'nan']
			labels = {"targets":meta, "corpus":state}

			# Extremely global
#			f = utils.word_freq_over_time(meta["year"], meta["lemmas"], color)
#			plt.savefig(f'{path_project}/lemmas_over_time.png', dpi=300, bbox_inches='tight')
#			plt.close('all')
#
#			# Global
#			exec('try:os.mkdir(out := os.path.join(path_analysis))\nexcept:pass')
#			f = utils.convergence_plot(lp, color, xticks=np.arange(0,10000,10))
#			plt.savefig(f'{out}/logdensity.png', dpi=300, bbox_inches='tight')
#			plt.close('all')

			def test():
				for label, dataset in labels.items():				
					# Dataset
					exec('try:os.mkdir(out := os.path.join(path_analysis, label))\nexcept:pass')
					topicprops = utils.topic_props_over_time_table(time=dataset["year"], z=dataset["z"], periods=periods)
					topicprops.to_csv(f'{out}/topic_props_over_time.csv')
					
					# Dataset, gender
					exec('try:os.mkdir(out := os.path.join(path_analysis, label, "gender"))\nexcept:pass')
					for gender in genders:
						topicprops = utils.topic_props_over_time_table(\
							time=dataset["year"], z=dataset["z"], periods=periods,\
							meta=dataset, variable='gender', value=gender)
						topicprops.to_csv(f'{out}/topic_props_over_time_{gender}.csv')

					# Dataset, party
					exec('try:os.mkdir(out := os.path.join(path_analysis, label, "party"))\nexcept:pass')
					for party in parties:
						topicprops = utils.topic_props_over_time_table(\
							time=dataset["year"], z=dataset["z"], periods=periods,\
							meta=dataset, variable='party_abbrev', value=party)
						topicprops.to_csv(f'{out}/topic_props_over_time_{party}.csv')

					# Dataset, lemma
					if label == "targets":
						exec('try:os.mkdir(out := os.path.join(path_analysis, label, "lemmas"))\nexcept:pass')
						for lemma in lemmas:
							data = dataset.loc[dataset["lemmas"] == lemma]
							exec('try:os.mkdir(out := os.path.join(path_analysis, label, "lemmas", lemma))\nexcept:pass')
							topicprops = utils.topic_props_over_time_table(time=data["year"], z=data["z"], periods=periods)
							topicprops.to_csv(f'{out}/topic_props_over_time.csv')

							# Dataset, lemma, gender
							exec('try:os.mkdir(out := os.path.join(path_analysis, label, "lemmas", lemma, "gender"))\nexcept:pass')
							for gender in genders:
								topicprops = utils.topic_props_over_time_table(\
									time=data["year"], z=data["z"], periods=periods,\
									meta=data, variable='gender', value=gender)
								topicprops.to_csv(f'{out}/topic_props_over_time_{gender}.csv')

							# Dataset, lemma, party
							exec('try:os.mkdir(out := os.path.join(path_analysis, label, "lemmas", lemma, "party"))\nexcept:pass')
							for party in parties:
								topicprops = utils.topic_props_over_time_table(\
									time=data["year"], z=data["z"], periods=periods,\
									meta=data, variable='party_abbrev', value=party)
								topicprops.to_csv(f'{out}/topic_props_over_time_{party}.csv')
							
				print(f'Project: {p}, window: {c}, topic: {k} finished after {round(time.time()-start)} seconds.')





