"""
NOTE: NOT FINISHED, BUT SOMETHING LIKE THIS
"""


bmw-meta.pydef rare_words():
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

	mop = pd.read_csv('../riksdagen-corpus/corpus/members_of_parliament.csv')

for p in ['j', 'f']:
	with open('data/target-words.json', 'r') as f:
		target_words = json.load(f)["target_" + p]
	for c in [5]:
		for k in [5]:
			# Set paths
			path_model = f'results/lda-models/{p}/window_{c}_topic_{k}/model'
			path_data = f'data/context_windows/{p}_window_{c}.csv'
			path_state = os.path.join(path_model, 'state.gz')
			state = state_to_df(path_state)
			M = max(state["#doc"])+1
			data = pd.read_csv(os.path.join(path_model, 'data.csv')) # name this clearer

			#if c == 5 and (k==20 or k==50):
			#	continue
			data["lemmas"] = list(map(lambda x: x[:4], data["target"]))
			data["gender"] = pd.Series(dtype=str)
			data["party_abbrev"] = pd.Series(dtype=str)
			for i, row in data.iterrows():
				idx = row["mop_id"]
				member = mop.loc[mop["id"] == idx]
				g = member["gender"].values
				
				party_abbrev = member["party_abbrev"].values
				if g.size > 0:
					data.loc[i,"gender"] = g
				if party_abbrev.size > 0:
					data.loc[i,"party_abbrev"] = party_abbrev
			data.to_csv(f'results/lda-models/{p}/meta.csv', index=False)