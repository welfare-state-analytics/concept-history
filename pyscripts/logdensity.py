import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

in_path = 'results/second-run'
folders = [f for f in [os.path.join(in_path, f) for f in os.listdir(in_path)] if os.path.isdir(f)]
results = folders[0]

logdensity = np.load(os.path.join(results, 'logdensity.npy'))

color = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1',\
                 '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']

epochs = 2000
burn_in = 1000

def convergence_plot(loss, burn_in=0, color=None):
	epochs = len(loss)
	df = pd.DataFrame({
		'epoch': list(range(epochs)), 
	    'y': loss,
	    'burn': ['A']*burn_in + ['B']*(epochs-burn_in)})
	
	pal = sns.color_palette(['#808080', color[0]]) if color is not None else color
	with sns.axes_style("whitegrid"):
		f = sns.lineplot(x='epoch', y='y', hue='burn', data=df, palette=pal)
		f.get_legend().remove()
		f.set_ylabel('')
		f.set(xlabel = "Epoch", ylabel = "Log joint density")
		sns.despine()

	return f

f = convergence_plot(logdensity, 1000, color)
plt.show()
plt.close()




