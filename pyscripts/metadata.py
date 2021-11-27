"""
Generates metadata files for context window datasets.
"""
import numpy as np
import pandas as pd
import os,json
from progressbar import progressbar
import re

def main():
	mop = pd.read_csv('../riksdagen-corpus/corpus/members_of_parliament.csv')
	variables = ["party_abbrev", "gender", "born", "id"]

	# Find files with smallest window sizes for all projects
	path = 'data/context_windows'
	files = [f for f in os.listdir(path) if f.endswith('json')]
	p = [f[0] for f in files]
	c = [int(re.search(r'\d+', f).group(0)) for f in files]
	df = pd.DataFrame({"c":c, "p":p}).groupby("p")["c"].min()

	for p,c in df.iteritems():
		file = f'{p}_window_{c}.json'
		with open(os.path.join(path, file)) as f:
		    idx = json.load(f)["id"]
		
		dt = dict(mop[variables].dtypes)
		meta = pd.DataFrame({c: pd.Series([np.nan]*len(idx), dtype=t) for c, t in dt.items()})

		for i in progressbar(range(len(idx))):
			member = mop[variables].values[mop["id"] == idx[i]]
			if len(member) > 0:
				meta.loc[i] = member[0]

		meta.to_csv(os.path.join(path, f'{p}_meta.csv'), index=False)

if __name__ == "__main__":
	main()