from math import log
from collections import Counter
from collections import defaultdict

from datetime import datetime

def entropy(probs):
	pc = 0.0
	for p in probs:
		if p != 0:
			pc -= p * log(p, 2)
	return pc

def information_gain(documents, word):
	'''
	for each document, [content, class]
	G(t) = -sum(P(c)*logP(c)) + P(t) * sum(P(c|t) * log(P(c|t))) + P(t) * sum(P(c|^t) * log(P(c|^t)))
	'''
	# print('--- calculate word: ##%s## information gain ---' % word)
	# length, classes, p_t = len(documents), Counter([cls for ctn, cls in documents]), 0.0
	# probs = [float(count)/length for c, count in classes.items()]
	# p1 = entropy(probs)
	ct1s, ct2s, t1, t2 = defaultdict(float), defaultdict(float), 0.0, 0.0
	for ctn, cls in documents:
		if word in ctn:
			t1 += 1
			ct1s[cls] += 1
		else:
			t2 += 1
			ct2s[cls] += 1
	probs1 = [n / t1 for k, n in ct1s.items()]
	probs2 = [n / t2 for k, n in ct2s.items()]
	p2, p3 = entropy(probs1), entropy(probs2)
	return - t1 * p2 - t2 * p3

def document_frequency(documents, word):
	'''
	assume the documents is [(ctn, cls), (ctn, cls)]
	DF(t) = # of t appears / # of documents
	'''
	return float(len([1 for ctn, cls in documents if word in ctn])) / len(documents)

def mutual_information(documents, word, t='MAX'):
	'''
	I(t) = sum(P(c) * log(P(t & c) / (P(t) * P(c))))
        I(t) = max(log(P(t & c)/ (P(t) * P(c))))
	A = # of times word and cls co-occur
	B = # of times word occurs 
	t = MAX, AVG
	'''
	# ts = {'MAX': max, 'AVG': AVG}
	length, classes = len(documents), Counter([cls for ctn, cls in documents])
	probs_c = {k:float(v)/length for k, v in classes.items()}
	As, B = defaultdict(lambda: 1e-10), 0.0
	for ctn, cls in documents:
		if word in ctn:
			As[cls] += 1
			B += 1
	if t == 'MAX':
		I_ct = max([log(As[c] / B / probs_c[c]) for c in probs_c.keys()])	
	elif t == 'AVG':
		I_ct = sum([(probs_c[c] * log(As[c] / B / probs_c[c])) for c in probs_c.keys()])
	return I_ct

def chi(documents, word, t='MAX'):
	'''
	A = # of times word and cls co-occurs
	B = # of times word occurs without cls
	C = # of times cls occurs without word
	D = # of times occurs neither word nor cls
	N = # of documents
	X^2 = N * (AD - BC)**2 / ( (A+C) * (B+D) * (A+B) * (C+D) )
	'''
	As, Cs, N = defaultdict(float), defaultdict(float), len(documents)
	for ctn, cls  in documents:
		if word in ctn:
			As[cls] += 1
		else:
			Cs[cls] += 1
	word_sum = sum([v for k, v in As.items()])
	Bs = {k: (word_sum - v) for k, v in As.items()}
	Ds = {k: (N - word_sum - v) for k, v in Cs.items()}
	Pcs = {k: (As[k] + Cs[k]) / N for k in As.keys()}
	Xs = {}
	for k in As.keys():
		A, B, C, D = As[k], Bs[k], Cs[k], Ds[k]
		Xs[k] = (A * D - B * C)**2 / ( (A+C) * (B+D) * (A+B) * (C+D) )
	if t == 'MAX':
		return max([v for k,v in Xs.items()])
	elif t == 'AVG':
		return sum([Pcs[k] * v for k,v in Xs.items()])

class Logging():
	def __init__(self, filepath='/home/lab/zyd/Documents/amazon/log/log1', timeformat=' '):
		self.fp = open(filepath, 'a')
		self.format = timeformat
		self.info('##### START A NEW LOG #####\n')

	def info(self, content):
		self.fp.write('------ %s ------\n' % (datetime.today().isoformat(self.format)))
		self.fp.write(content + '\n\n')

if __name__ == '__main__':
	# documents = [([''],1)] * 42 + [(['good'],1)] * 6 + [(['good'],0)] * 4 + [(['excellent'],1)] * 2 + [(['excellent'],0)] + [([''],0)] * 45
	documents = [(['good'],'c1')] * 10 + [(['good'],'c2')] * 20 + [(['good'],'c3')] * 30 + [([''],'c1')] * 10 + [([''],'c2')] * 10 + [([''],'c3')] * 20
	# words = ['good', 'excellent']
	# words = ['good']
	# for w in words:
		# print(information_gain(documents, w))
		# print(chi(documents, w))

	documents = [(['a', 'b', 'c'], 0),(['a', 'b'], 1),(['a', 'c'], 1),(['e', 'b', 'c'], 1),(['a', 'd', 'e'], 1)]
	print(document_frequency(documents, 'a'))
	# print(mutual_information(documents, 'a'))

