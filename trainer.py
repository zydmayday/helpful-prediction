'''
given training data to output a trained model
'''
import nltk

class Trainer:
	def __init__(self, classifier=None, train_data=None):
		if classifier:
			self.data = train_data
		if train_data:
			self.classifier = classifier

	def train(self, train_data=None):
		if train_data:
			self.data = train_data
		classifier = self.classifier.train(self.data)
		return classifier

	def set_data(self, train_data):
		self.data = train_data

if __name__ == '__main__':
	pass
