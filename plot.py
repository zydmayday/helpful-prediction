'''
provide the methods of how to plot the data into graph
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pre_process import Preprocess
from collections import Counter 
import sys
sys.path.insert(0, '/home/lab/zyd/Documents/amazon/lib')
from plot_learning_curve import plot_learning_curve
import numpy as np
from database import AmazonDB
plt.ioff()


class Plot:

	def __init__(self):
		pass

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

	def cross_validation(self, estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), savename=''):
		plt = plot_learning_curve(estimator, title, X, y, ylim=ylim, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
		plt.savefig(savename)

def review_word_count_hist(gt=100):
	db = AmazonDB()
	pp = Preprocess()
	reviews = db.get_all(cols=['content'], gt=gt)
	count_list = []
	for r in reviews:
		count_list.append(len(pp.tokenize(r[0])))
	length = len(set(count_list))
	count_list = [v for v in count_list if v <= length]
	plt.yscale('log', nonposy='clip')
	plt.title('count review word / %d' % gt)
	plt.ylabel('count')
	plt.xlabel('review length')
	plt.hist(count_list, length)
	plt.savefig('/home/lab/zyd/Documents/amazon/images/review_word_count_hist_%s_over%d.png' % (length, gt))

def helpful_count_hist(gt=100):
	db = AmazonDB()
	pp = Preprocess()
	reviews = db.get_all(cols=['helpful'], gt=gt)
	count_list = []
	for r in reviews:
		count_list.append(r[0])
	length = len(set(count_list))
	count_list = [v for v in count_list if v <= length]
	plt.yscale('log', nonposy='clip')
	plt.title('count helpful sum / %d' % gt)
	plt.xlabel('helpful')
	plt.ylabel('helpful sum')
	plt.hist(count_list, length)
	plt.savefig('/home/lab/zyd/Documents/amazon/images/helpful_count_hist_%s_over%d.png' % (length, gt))


if __name__ == '__main__':
	# review_word_count_hist()
	helpful_count_hist()
	# plot = Plot()
	# pp = Preprocess()
	# ph_list = pp.get_percent_helpful()
	# for i, v in enumerate(ph_list):
	# 	ph_list[i] = round(v, 2)  
	# count = Counter(ph_list)
	# get data from database
	

