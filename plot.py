'''
provide the methods of how to plot the data into graph
'''
import matplotlib.pyplot as plt
from pre_process import Preprocess
from collections import Counter 

class Plot:

	def histogram(self, x, xlabel='x label', ylabel='y label', title='title'):
		# the histogram of the data
		n, bins, patches = plt.hist(x, 50, normed=1, facecolor='g', alpha=0.75)

		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.title(title)
		plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
		plt.axis([min(x), max(x), 0, 0.03])
		plt.grid(True)
		plt.show()

if __name__ == '__main__':
	plot = Plot()
	pp = Preprocess()
	ph_list = pp.get_percent_helpful()
	for i, v in enumerate(ph_list):
		ph_list[i] = round(v, 2)  
	count = Counter(ph_list)
	# get data from database
	

