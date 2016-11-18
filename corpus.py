'''
we will then put all of data into mongodb,
but first since we just have a mysql database, so if we want to save list data,
we should pickle/dump them on the disk,
and also we have some raw text file to deal with.
so we need this module to handle such problem until we can put all of the data into mongodb.
'''
from nltk.corpus import PlaintextCorpusReader
from database import AmazonDB
import nltk
import pickle

class Corpus:

	def __init__(self, corpus_root='', dbpath=''):
		if corpus_root:
			self.wordlists = PlaintextCorpusReader(corpus_root, '.*.corp')
		if dbpath:
			self.db = AmazonDB(dbpath)

	def get_corpus(self, name='reviews.corp'):
		return self.wordlists.words(name)

	def save_corpus_by_rating(self):
		sql = '''SELECT RATING,CONTENT FROM REVIEW where rating=1 or rating=5;'''
		reviews = self.db.execute(sql)
		corpus_rating1 = ""
		corpus_rating5 = ""
		for r in reviews:
			if r[0] == 1:
				corpus_rating1 += r[1]
			elif r[0] == 5:
				corpus_rating5 += r[1]
		fw = open('/home/data/amazon/reviews_rating1.corp', 'w')
		fw.write(corpus_rating1)
		fw = open('/home/data/amazon/reviews_rating5.corp', 'w')
		fw.write(corpus_rating5)
		fw.close()

	def get_X(self):
		'''
		get the input 
		'''


	def _save_fdist(self, name='reviews.corp', save_path='/home/data/amazon/reviews_fdist.pkl'):
		words = self.get_corpus(name)
		words = self._remove_stopwords(words)
		words = self._remove_punctuation(words)
		fdist = nltk.FreqDist(words)
		pickle.dump(fdist, open(save_path, 'wb'))

if __name__ == '__main__':
	cor = Corpus(dbpath='/home/lab/zyd/Documents/amazon/amazon.db')
	# cor.save_corpus_by_rating()
	cor._save_fdist()
