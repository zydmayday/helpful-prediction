'''
we can just get the raw data or some cleaned data from database,
but the cleaned data will also be pre precessed by this module's function.
so we need to use this module's function to transform the data to the form we need.
it kinds like a preparation work before training.
'''
from database import AmazonDB
from corpus import Corpus
import nltk
import pickle
from nltk.tokenize import RegexpTokenizer
import utils
import time
from multi_process import feature_multi_process, filter_multi_process

class Preprocess:
	def tokenize(self, corpus, tokenizer=RegexpTokenizer('\w+'), pre_filter=True):
		'''
		determine how to tokenize the corpus
		'''
		tokens = tokenizer.tokenize(corpus)
		if pre_filter:
			tokens = self.remove_digit(self.remove_by_length(self.remove_stopwords(self.remove_punctuation(tokens))))
		return tokens

	def word_features(self, documents, size=4000, method='DF', threshhold=0.001, use_multi=True):
		'''
		determine the features of train data
		rtype: (list[{str: int}], int)
		'''
		methods = {'DF': utils.document_frequency, 'IG': utils.information_gain, 'MI': utils.mutual_information, 'CHI': utils.chi}
		words = set()
		for ctn, cls in documents:
			for w in ctn:
				words.add(w)
		print('------ we have %d unique words ------' % len(words))
		fdist = {}
		if use_multi:
			fdist = feature_multi_process(documents, words, methods[method])
		else:
			for w in words:
				fdist[w] = methods[method](documents, w)
		wf = sorted(fdist.items(), key=lambda i:i[1], reverse=True)
		if size == -1:
			return wf, len(words)
		else:
			return wf[:size], len(words)

	def document_features(self, document, word_features):
		'''
		determine every document's feature vector
		'''
		document_words = set(document)
		features = [0] * len(word_features)
		for idx, wf in enumerate(word_features):
			if wf[0] in document_words:
				features[idx] = 1
		return features

	def class_features(self, c, threshhold=45):
		'''
		determine the threshhold of class label
		for binary classification, g means good, b means bad
		'''
		if c < threshhold:
			return 0
		else:
			return 1

	def remove_stopwords(self, words):
		stopwords = nltk.corpus.stopwords.words('english')
		content = [w for w in words if w.lower() not in stopwords]
		return content

	def remove_punctuation(self, words):
		from string import punctuation
		content = [w for w in words if w.lower() not in punctuation]
		return content

	def remove_by_length(self, words, length=range(2,11)):
		content = [w for w in words if len(w) in length]
		return content		

	def remove_digit(self, words):
		content = [w for w in words if not w.isdigit()]
		return content

	def filter(self, documents, wfk, use_multi=True):
		'''
		get word in document where it is in word_features
		return string type
		'''
		# wfk = [k[0] for k in word_features]
		if use_multi:
			return filter_multi_process(documents, wfk)


if __name__ == '__main__':
	for m in ['DF','IG', 'MI', 'CHI']:
		start_time = time.time()
		db = AmazonDB()
		pp = Preprocess()
		reviews = [(r,h) for r,h in db.get_all(cols=['content', 'helpful'], gt=10)][:100]
		documents = [(pp.tokenize(ctn), pp.class_features(helpful)) for ctn, helpful in reviews]
		# word_features1 = pp.word_features(documents, method='DF')
		word_features = pp.word_features(documents, method=m)[:20]
		print(word_features)
	# word_features3 = pp.word_features(documents, method='MI')
