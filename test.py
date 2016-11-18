'''
given the classifier or trained model,
use test data to test the performance/precision of model
'''

from trainer import Trainer
from pre_process import Preprocess
from database import AmazonDB
from plot import Plot
import nltk
import time

from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.svm import SVC

import numpy as np
from sklearn.model_selection import ShuffleSplit

import random

from sklearn.model_selection import KFold, cross_val_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from utils import Logging


class Tester:

	def __init__(self, classifier=None, test_data=None):
		if classifier:
			self.classifier = classifier
		if test_data:
			self.test_data = test_data

	def test(self, mif_num=10):
		classifier = self.classifier
		accuracy = nltk.classify.accuracy(classifier, self.test_data)
		most_informative_features = classifier.show_most_informative_features(mif_num)
		return accuracy, most_informative_features

	def cross_validation(self, data, fold=10):
		X = [x for x,y in data]
		Y = [y for x,y in data]
		scores = cross_val_score(self.classifier, X, Y, cv=fold)
		m, std = scores.mean(), scores.std() * 2
		print("Accuracy: %0.2f (+/- %0.2f)" % (m, std))
		return scores, m, std

def main():
	db = AmazonDB()
	reviews = [(r,h) for r,h in db.get_all(cols=['content', 'helpful'], gt=100)]
	random.shuffle(reviews)
	reviews = reviews[:10000]
	print("--- totally got %d reviews ---" % (len(reviews)))
	start_time = time.time()
	print("---%d--- get raw data --- %s seconds ---" % (loop, time.time() - start_time))
	pp = Preprocess()
	documents = [(pp.tokenize(ctn), pp.class_features(helpful)) for ctn, helpful in reviews]
	print("---%d--- get documents --- %s seconds ---" % (loop, time.time() - start_time))
	word_features_size = 1000
	d_dist = {'IG':[], 'DF': []}
	for m in m_dist.keys():
		word_features = pp.word_features(documents, size=word_features_size, method=m)
		data_set = [(pp.document_features(r, word_features), h) for (r,h) in documents]
		d_dist[m] = data_set
	print("---%d--- get dataset --- %s seconds ---%s---" % (loop, time.time() - start_time, m))
	for loop in range(200, word_features_size + 1, 200):
		
		X, Y = [x for x,y in data_set], [y for x,y in data_set]
		classifier, plot = GaussianNB(), Plot()
		plot.cross_validation(classifier, "naive bayes / 10000 reviews / %d / %s" % (loop, m), X, Y, 
			cv=ShuffleSplit(n_splits=100, test_size=0.1, random_state=0), train_sizes=np.linspace(.1, 1.0, 5), 
			n_jobs=12, savename='/home/lab/zyd/Documents/amazon/images/cv%d_%s.png' % (loop, m), ylim=(0.3, 1.01))
		print("---%d--- train test plot --- %s seconds ---%s---" % (loop, time.time() - start_time, m))

def cross_validation_main(review_sizes=[], word_features_size = 4000, gap=2000):
	db = AmazonDB()
	reviews = [(r,h) for r,h in db.get_all(cols=['content', 'helpful'], gt=100)]
	count_vect = CountVectorizer()
	for review_size in review_sizes:
		random.shuffle(reviews)
		rs = reviews[:review_size]
		print('reviews size: %d' % (len(rs)))

		start_time = time.time()
		print("------ get raw data --- %s seconds ---" % (time.time() - start_time))
		pp = Preprocess()
		documents = [(pp.tokenize(ctn), pp.class_features(helpful)) for ctn, helpful in rs]
		print("------ get documents --- %s seconds ---" % (time.time() - start_time))

		d_dist = {'IG':['o-', 'g'], 'DF': ['o-', 'r'], 'MI': ['o-', 'b'], 'CHI': ['o-', 'k']}
		for m in d_dist.keys():
			stt = time.time()
			word_features = pp.word_features(documents, size=word_features_size, method=m)
			print("------ get word features --- %s seconds ---%s---" % (time.time() - stt, m))
			train_x = count_vect.fit_transform([pp.filter(d, word_features) for d, c in documents])
			targets = [c for d, c in documents]
			# data_set = [(pp.document_features(r, word_features), h) for (r,h) in documents]
			print("------ get dataset from word freatures --- %s seconds ---%s---" % (time.time() - stt, m))
			scores_list = []
			size_list = [gap * i for i in range(1, int(word_features_size / gap) + 1)]
			for size in size_list:
				'''
				here we define a parameter called size, for that we could determine the dimention of our input.
				use 10-fold cross validation to test the precision of our model.

				'''
				# X, Y = [x[:size] for x,y in data_set], [y for x,y in data_set]
				X, Y = [x[:size] for x in train_x], targets
				k_fold, classifier = KFold(n_splits=10), MultinomialNB()
				scores = cross_val_score(classifier, X, Y, cv=k_fold, n_jobs=-1)
				scores_list.append(scores)
				print("---%s---%d--- append test scores into list, %s" % (m, size, scores))
			test_scores_mean = np.mean(scores_list, axis=1)
			test_scores_std = np.std(scores_list, axis=1)

			plt.title("comparison for feature selections / %s" % review_size)
			plt.ylim((0.3, 1.01))
			plt.xlabel("dimention")
			plt.ylabel("Score")
			plt.grid()
			plt.fill_between(size_list, test_scores_mean - test_scores_std,
							test_scores_mean + test_scores_std, alpha=0.1, color=d_dist[m][1])
			plt.plot(size_list, test_scores_mean, d_dist[m][0], color=d_dist[m][1], label=m)
			plt.legend(loc="best")
		print('------ save cross_validation_size%d.png' % (review_size))
		plt.savefig('images/cross_validation_size%d.png' % (review_size))
		plt.clf()

def cross_validation_main2(review_sizes=[], word_features_sizes = [], feature_type=['0or1', 'count', 'TDIDF'], classifier=MultinomialNB(), classfier_name='naive_bayes'):
	log = Logging()
	db = AmazonDB()
	# reviews = [(r,h) for r,h in db.get_all(cols=['content', 'helpful'], gt=100)]
	reviews_good = [(r,h) for r,h in db.get_content_helpful(cols=['content', 'helpful'], gt=30)]
	reviews_bad = [(r,h) for r,h in db.get_content_helpful(cols=['content', 'helpful'], lt=1)]
	count_vect = CountVectorizer()
	tfidf_transformer = TfidfTransformer()

	for review_size in review_sizes:
		log.info("------ setting review size(%d) feature size(%d) ------" % (review_size, word_features_sizes[-1]))
		random.shuffle(reviews_good)
		random.shuffle(reviews_bad)
		rs = reviews_good[:int(review_size/2)] + reviews_bad[:int(review_size/2)]
		random.shuffle(rs)
		start_time = time.time()
		log.info("------ get raw data --- %.3f seconds ---" % (time.time() - start_time))
		pp = Preprocess()
		documents = [(pp.tokenize(ctn), pp.class_features(helpful, threshhold=20)) for ctn, helpful in rs]
		log.info("------ get documents --- %.3f seconds ---" % (time.time() - start_time))
		d_dist = {'IG':['o-', 'g'], 'DF': ['o-', 'r'], 'MI': ['o-', 'b'], 'CHI': ['o-', 'k']}
		for m in d_dist.keys():
			stt = time.time()
			scores_list = []
			# get all words satisfied the threshhold, with the max size
			word_features, word_len = pp.word_features(documents, size=-1, method=m)
			wfk = [k[0] for k in word_features]
			log.info(" ---%s--- get word features count(%d)--- %.3f seconds" % (m, len(word_features) ,time.time() - stt))
			for word_features_size in word_features_sizes:
				stt = time.time()
				# filter the documents with specific size of features
				word_features_size = int(word_features_size * word_len)
				X, Y = None, None
				if feature_type == '0or1':
					X = [pp.document_features(d, word_features[:word_features_size]) for d, c in documents]
					Y = [c for d, c in documents]
				elif feature_type == 'count':
					filtered_docs = pp.filter(documents, wfk[:word_features_size])
					X = count_vect.fit_transform([d for d, c in filtered_docs])
					Y = [c for d, c in filtered_docs]
				elif feature_type == 'TDIDF':
					filtered_docs = pp.filter(documents, wfk[:word_features_size])
					X = count_vect.fit_transform([d for d, c in filtered_docs])
					X = tfidf_transformer.fit_transform(X)
					Y = [c for d, c in filtered_docs]
				
				log.info("---%s--- filter dataset by word features with size(%d)  --- %.3f seconds " % (m, word_features_size, time.time() - stt))
				k_fold = KFold(n_splits=10)
				scores = cross_val_score(classifier, X, Y, cv=k_fold, n_jobs=-1)
				log.info("---%s--- filter dataset by word features with size(%d)  --- %.3f seconds " % (m, word_features_size, time.time() - stt))
				scores_list.append(scores)
				# print("---%s---%d--- append test scores into list, %s" % (m, word_features_size, scores))
			test_scores_mean = np.mean(scores_list, axis=1)
			test_scores_std = np.std(scores_list, axis=1)

			log.info("---%s--- scores list %s ---%d--- " % (m, test_scores_mean, review_size))
			print("---%s--- scores list %s ---%d--- " % (m, test_scores_mean, review_size))

			plt.title("cross validation / %s / %s / %s" % (classfier_name, review_size, feature_type))
			plt.ylim((0.3, 1.01))
			plt.xlabel("dimension")
			plt.ylabel("Score")
			plt.grid()
			plt.fill_between(word_features_sizes, test_scores_mean - test_scores_std,
							test_scores_mean + test_scores_std, alpha=0.1, color=d_dist[m][1])
			plt.plot(word_features_sizes, test_scores_mean, d_dist[m][0], color=d_dist[m][1], label=m)
			plt.legend(loc="best")
		log.info('------ save cross_validation_%s_size%d_%s_feature_type(%s).png' % (classfier_name, review_size, word_len, feature_type))
		plt.savefig('images/%s/cross_validation_%s_size%d_%s_feature_type(%s).png' % (classfier_name, classfier_name, review_size, word_len, feature_type))
		plt.clf()


if __name__ == '__main__':
	'''
	if we use too small feature size, 
	we will almostly remove all words from each review,
	which means every review will have just the same input vector,
	which will cause a very bad prediction result.
	'''
	# cross_validation_main2([2000, 18000], [i*0.05 for i in range(1, 21)], feature_type='count', classifier=SVC(), classfier_name='SVC')
	cross_validation_main2([2000, 18000], [i*0.05 for i in range(1, 21)], feature_type='count', classifier=MultinomialNB(), classfier_name='NaiveBayes')
	# cross_validation_main2([2000], [i*0.05 for i in range(1, 21)], feature_type='count', classifier=RandomForestClassifier(), classfier_name='RandomForest')
	# cross_validation_main2([10], [1], feature_type='count')
	# cross_validation_main2([50000], [i*0.05 for i in range(1, 21)], feature_type='0or1')
	# cross_validation_main2([50000], [i*0.05 for i in range(1, 21)], feature_type='count')
	# cross_validation_main2([50000], [i*0.05 for i in range(1, 21)], feature_type='TDIDF')
	# cross_validation_main2([10000], [i*20 for i in range(1,2)])