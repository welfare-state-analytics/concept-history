"""
Make context_windows.py scrape the metadata and return it as a df which is used here.
Probably crashes atm due to mop_id being talman or mop_sk id, etc.
"""

import numpy as np
import pandas as pd
import os, json
from pygibbs import utils
import matplotlib.pyplot as plt
import time
from collections import Counter
import gzip
import sklearn.preprocessing

def state_to_df(statefile):
    """Transform state file into pandas dataframe.
    The MALLET statefile is tab-separated, and the first two rows contain the alpha and beta hypterparamters.
    
    Args:
        statefile (str): Path to statefile produced by MALLET.
    Returns:
        datframe: topic assignment for each token in each document of the model
    """
    return pd.read_csv(statefile,
                       compression='gzip',
                       sep=' ',
                       skiprows=[1,2]
                       )

# Load global data
folder = 'lda-models'
mop = pd.read_csv('../riksdagen-corpus/corpus/members_of_parliament.csv')
with open('../riksdagen-corpus/corpus/party_mapping.json', 'r') as f:
	party_map = json.load(f)
with open('data/stopwords-sv.txt','r') as f:
	stopwords = f.read().splitlines()
with open('data/stopwords-sv.txt','r') as f:
	stopwords = f.read().splitlines()
periods = np.array([list(np.arange(1920+(10*i), 1920+(10*(i+1)))) for i in range(11)])

alpha, beta = 0.5, 0.5
project = ['j', 'f']
window_sizes = [5, 10, 20]
n_topics = [5, 10, 20, 50]

p, c, k = 'j', 5, 5
for p in project:
	for c in window_sizes:
		for k in n_topics:
			# Skips unused combinations
			if c == 5 and (k==20 or k==50):
				continue

			start = time.time()
			# Set paths
			results_path = f'results/{folder}/{p}/window_{c}_topic_{k}/model'
			path_to_data = f'data/context_windows/{p}_window_{c}.csv'
			path_state = os.path.join(results_path, 'state.gz')
			out = f'results/{folder}/{p}/window_{c}_topic_{k}/analysis'
			exec('try:os.mkdir(out)\nexcept:pass')

			# Metadata
			meta = pd.read_csv(path_to_data)
			meta["lemmas"] = list(map(lambda x: x[:4], meta["target"]))
			meta["gender"] = pd.Series(dtype=str)
			meta["party_abbrev"] = pd.Series(dtype=str)
			for i, row in meta.iterrows():
				idx = row["mop_id"]
				member = mop.loc[mop["id"] == idx]
				g = member["gender"].values
				
				party_abbrev = member["party_abbrev"].values
				if g.size > 0:
					meta.loc[i,"gender"] = g
				if party_abbrev.size > 0:
					meta.loc[i,"party_abbrev"] = party_abbrev

			# Extended metadata
			state = state_to_df(path_state)
			data = []
			N = Counter(state["#doc"])
			for i, row in meta.iterrows():
				data.extend([row.tolist()]*N[i])
			meta_big = pd.DataFrame(data, columns = meta.columns)
			meta_big["z"] = state["topic"]

			# Load vocab
			with open(os.path.join(results_path, 'vocab.json'), 'r') as f:
				vocab = json.load(f)
			V = len(vocab)

			if k <= 10:
				color = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1',\
						 '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF'][:k]
			else:
				color = None

			# Load results
			lp 		= np.load(os.path.join(results_path, 'logperplexity.npy'))
			z 		= np.load(os.path.join(results_path, 'z.npy'))
			theta 	= np.load(os.path.join(results_path, 'theta.npy'))
			phi 	= np.load(os.path.join(results_path, 'phi.npy'))
			Nd 		= np.load(os.path.join(results_path, 'Nd.npy'))
			Nk 		= np.load(os.path.join(results_path, 'Nk.npy'))

			# Sort all topics in target word frequency order
			z, theta, phi, Nd, Nk, z_sorted = utils.z_sorter(z, theta, phi, Nd, Nk)
			for i in range(len(meta_big)):
				meta_big.loc[i,"z"] = z_sorted[meta_big.loc[i,"z"]]
			meta["z"] = z
			M = len(meta)

			#meta = pd.read_csv(os.path.join(results_path, 'meta.csv'))
			#meta_big = pd.read_csv(os.path.join(results_path, 'meta_big.csv'))

			parties = [p for p in list(set(meta["party_abbrev"])) if str(p) != 'nan']

			def test():
				# Global
				f = utils.convergence_plot(lp, color, xticks=np.arange(0,10000,10))
				plt.savefig(os.path.join(out, 'logdensity.png'), dpi=300, bbox_inches='tight')
				plt.close('all')

				f = utils.word_freq_over_time(meta["year"], meta["lemmas"], color)
				plt.savefig(os.path.join(out, 'lemmas_over_time.png'), dpi=300, bbox_inches='tight')
				plt.close('all')

				topwords = utils.topWords(phi, vocab, k, stopwords, list(set(meta["target"])))
				topwords.to_csv(os.path.join(out, 'top_words_targets_removed.csv'), index=False)

				distinctwords = utils.distinctWords(Nk, vocab, k)
				distinctwords.to_csv(os.path.join(out, 'distinct_words.csv'), index=False)

				lemmaprops = utils.lemma_props_by_topic(meta["lemmas"], meta["z"])
				lemmaprops.to_csv(os.path.join(out, 'lemma_topic_distributions.csv'))

				topdocs = utils.lda_top_docs(meta, theta, n=50)
				with open(os.path.join(out, 'top_docs_by_topic.txt'), 'w') as f:
					for line in topdocs:
						f.write(line + '\n'*2)

				# Local
				labels = {"targets":meta, "corpus":meta_big}
				for label, dataset in labels.items():
					exec('try:os.mkdir(os.path.join(out, label))\nexcept:pass')
					f = utils.senses_over_time(dataset["year"], dataset["z"], color)
					plt.savefig(os.path.join(out, label, 'senses_over_time.png'), dpi=300, bbox_inches='tight')
					plt.close('all')
					
					topicprops = utils.topic_props_over_time_table(dataset["year"], dataset["z"], periods)
					topicprops.to_csv(os.path.join(out, label, 'topic_props_over_time.csv'))

					exec('try:os.mkdir(os.path.join(out, label, "gender"))\nexcept:pass')
					for gender in ["woman", "man"]:
						#Table
						topicpropsparty = utils.topic_props_over_time_table(dataset["year"], dataset["z"], periods, meta=dataset, variable="gender", value=gender)
						topicpropsparty.to_csv(os.path.join(out, f'{label}/gender/topic_props_over_time_{gender}.csv'))
								
						# Plot
						f = utils.senses_over_time(dataset["year"], dataset["z"], color, meta=dataset, variable="gender", value=gender)
						f.suptitle('Women' if gender == "woman" else "Men")
						gender_abbrev = "m" if gender == "man" else "f"
						plt.savefig(os.path.join(out, f'{label}/gender/senses_over_time_{gender_abbrev}.png'), dpi=300, bbox_inches='tight')
						plt.close('all')

					exec('try:os.mkdir(os.path.join(out, label, "party"))\nexcept:pass')
					for party in parties:
						# Table
						topicpropsparty = utils.topic_props_over_time_table(dataset["year"], dataset["z"], periods, meta=dataset, variable="party_abbrev", value=party)
						topicpropsparty.to_csv(os.path.join(out, f'{label}/party/topic_props_over_time_{party}.csv'))

						# Plot
						f = utils.senses_over_time(dataset["year"], dataset["z"], color, dataset, "party_abbrev", party, xlim=False)
						f.suptitle(max([key for key, value in party_map.items() if value == party]))
						plt.savefig(os.path.join(out, f'{label}/party/senses_over_time_{party}.png'), dpi=300, bbox_inches='tight')
						plt.close('all')

			#		exec('try:os.mkdir(os.path.join(out, "topics"))\nexcept:pass')
			#		for topic in range(k):
			#			f = utils.lemmas_over_time_by_topic(meta["lemmas"], meta["year"], z, topic)
			#			plt.savefig(os.path.join(out, f'topics/lemmas_over_time_by_topic_{topic}.png'), dpi=300, bbox_inches='tight')
			#			plt.close('all')


			test()
			print(f'Project: {p}, window: {c}, topic: {k} finished after {round(time.time()-start)} seconds.')