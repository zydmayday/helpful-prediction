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

class Preprocess:
	def __init__(self, database=AmazonDB(), corpus=Corpus()):
		self.db = database
		self.corp = corpus

	def get_product_helpful_dict(self):
		'''
		add productid is key, sum of helpful belong to productid as value
		'''
		result = self.db.get_helpful_group_by_product()
		ph_dict = {}
		for r in result:
			ph_dict[r[0]] = r[1]
		return ph_dict

	def save_X_Y(self, target_save_data=[], target_save_sql=""):
		'''
		'''
		ph_dict = self.get_product_helpful_dict()
		reviews = self.db.execute('''SELECT CONTENT,HELPFUL,PRODUCTID FROM REVIEW''')
		# fdist = nltk.FreqDist(self.corp.get_corpus('reviews.corp'))
		fdist = pickle.load(open('/home/data/amazon/reviews_fdist.pkl', 'rb'))
		for x, y, pid in reviews:
			X = self._get_features(fdist, nltk.word_tokenize(x))
			sum_pid = ph_dict.get(pid, 0)
			if sum_pid:
				Y = float(y) / sum_pid
			else:
				Y = 0
			sql = '''INSERT INTO DATASET_1 (X,Y) VALUES ("%s",%.2f);''' % (X, Y)
			print sql
			self.db.save(sql)

	def _get_features(self, fdist, tokens):
		return [fdist.get(t, 0) for t in tokens]

	def get_percent_helpful(self):
		ph_dict = self.get_product_helpful_dict()
		helpful_list = self.db.get_all_helpful()
		percent_hlist = []
		for h, pid in helpful_list:
			psum = ph_dict.get(pid, 0)
			if psum:
				percent_hlist.append(float(h) / psum)
			else:
				percent_hlist.append(0.0)
		return percent_hlist

if __name__ == '__main__':
	adb = AmazonDB()
	corp = Corpus()
	review = Preprocess(adb, corp)
	# ph_dict = review.get_product_helpful_dict()
	review.save_X_Y()