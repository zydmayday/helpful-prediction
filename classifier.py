import pickle
import nltk

class Classifier:
	def __init__(self, classifier=nltk.NaiveBayesClassifier):
		self.classifier = classifier

	def get(self):
		return classifier

	