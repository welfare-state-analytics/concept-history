"""
Generates metadata files for context window datasets.
"""
import numpy as np
import pandas as pd
import os,json
from progressbar import progressbar
import re
import argparse
import yaml

def main(args):
	with open(args.config, 'r') as f:
		config = yaml.safe_load(f)
	mop = pd.read_csv(os.path.join(config["paths"]["corpus"], 'members_of_parliament.csv'))

	# Find files with smallest window sizes for relevant projects
	files = [f for f in os.listdir(config["paths"]["data"]) if
			f.endswith('.json') and
			f[0 in config["projects"]] and
			int(re.findall(r'\d+', f)[0]) in config["window_sizes"]]

	p = [f[0] for f in files]
	c = [int(re.search(r'\d+', f).group(0)) for f in files]
	df = pd.DataFrame({"c":c, "p":p}).groupby("p")["c"].min()

	for p,c in df.iteritems():
		file = f'{p}_window_{c}.json'
		with open(os.path.join(config["paths"]["data"], file)) as f:
			idx = json.load(f)["id"]

		dt = dict(mop[config["metadata"]].dtypes)
		meta = pd.DataFrame({c: pd.Series([np.nan]*len(idx), dtype=t) for c, t in dt.items()})

		for i in progressbar(range(len(idx))):
			member = mop[config["metadata"]].values[mop["id"] == idx[i]]
			if len(member) > 0:
				meta.loc[i] = member[0]
		meta.to_csv(os.path.join(config["paths"]["data"], f'{p}_meta.csv'), index=False)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--config", type=str)
    args = argparser.parse_args()
    main(args)
