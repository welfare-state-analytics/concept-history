import numpy as np
import pandas as pd
import os, json
from pygibbs import utils
import matplotlib.pyplot as plt
import argparse
import yaml
from time import time

def main():
	p, c, k = 'j', 5, 5
	in_path = f'results/lda-models/{p}/window_{c}_topic_{k}/model'
	logperplexity = np.load(os.path.join(in_path, 'logperplexity.npy'))

	color = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1',\
	         '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF'][:k]

	f = utils.convergence_plot(logperplexity, 1, color)
	plt.show()
	#plt.savefig(os.path.join(out, 'logdensity.png'), dpi=300, bbox_inches='tight')
	plt.close('all')

if __name__ == '__main__':
    main()