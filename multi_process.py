import multiprocessing
import time
from utils import information_gain

class FeatureConsumer(multiprocessing.Process):
    
    def __init__(self, task_queue, result_dict):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_dict = result_dict

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                # print('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break
            # print('%s: %s' % (proc_name, next_task))
            answers = next_task.do()
            self.task_queue.task_done()
            for w, v in answers:
            	self.result_dict[w] = v
        return

class FilterConsumer(multiprocessing.Process):
    
    def __init__(self, task_queue, results):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.results = results

    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                # print('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break
            # print('%s: %s' % (proc_name, next_task))
            self.results += next_task.do()
            self.task_queue.task_done()
        return

class FeatureTask(object):
    def __init__(self, documents, words, method=information_gain):
        self.documents = documents
        self.words = words
        self.method = method

    def do(self):
    	rs = [(w, self.method(self.documents, w)) for w in self.words]
    	return rs

class FilterTask(object):
	def __init__(self, documents, wfk):
		self.documents = documents
		self.wfk = wfk

	def do(self):
		corps = []
		for document, cls in self.documents:
			corp = ' '.join([w for w in document if w in self.wfk])
			if not corp:
				corp = 'NOVALUEWORD'
			corps.append((corp, cls))
		return corps

def partition(lst, n):
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

def feature_multi_process(documents, words, method, task=FeatureTask, consumer=FeatureConsumer, jobs=-1):
	manager = multiprocessing.Manager()
	results = manager.dict()
	tasks = multiprocessing.JoinableQueue()

	# Start consumers
	num_consumers = jobs
	if jobs == -1:
		num_consumers = int(multiprocessing.cpu_count() / 2)
	# print('Creating %d consumers' % num_consumers)
	consumers = [ consumer(tasks, results)
	              for i in range(num_consumers) ]
	for w in consumers:
	    w.start()

	for ws in partition(list(words), num_consumers):
		tasks.put(task(documents, ws, method))

	# Add a poison pill for each consumer
	for i in range(num_consumers):
	    tasks.put(None)

	# Wait for all of the tasks to finish
	tasks.join()
	return results

def filter_multi_process(documents, wkf, task=FilterTask, consumer=FilterConsumer, jobs=-1):
	manager = multiprocessing.Manager()
	results = manager.list()
	tasks = multiprocessing.JoinableQueue()

	# Start consumers
	num_consumers = jobs
	if jobs == -1:
		num_consumers = int(multiprocessing.cpu_count() / 2)
	# print('Creating %d consumers' % num_consumers)
	consumers = [ consumer(tasks, results)
	              for i in range(num_consumers) ]
	for w in consumers:
	    w.start()

	for docs in partition(list(documents), num_consumers):
		tasks.put(task(docs, wkf))

	# Add a poison pill for each consumer
	for i in range(num_consumers):
	    tasks.put(None)

	# Wait for all of the tasks to finish
	tasks.join()
	return results

if __name__ == '__main__':
	start_time = time.time()
	# Establish communication queues
	tasks = multiprocessing.JoinableQueue()
	# results = multiprocessing.Queue()

	# Enqueue jobs
	from database import AmazonDB
	from pre_process import Preprocess
	db = AmazonDB()
	pp = Preprocess()
	reviews = [(r,h) for r,h in db.get_all(cols=['content', 'helpful'], gt=10)][:5000]
	documents = [(pp.tokenize(ctn), pp.class_features(helpful)) for ctn, helpful in reviews]
	words = set()
	for ctn, cls in documents:
		for w in ctn:
			words.add(w)
	
	manager = multiprocessing.Manager()
	word_features = manager.dict()

	# Start consumers
	num_consumers = int(multiprocessing.cpu_count() / 2)
	print('Creating %d consumers' % num_consumers)
	consumers = [ Consumer(tasks, word_features)
	              for i in range(num_consumers) ]
	for w in consumers:
	    w.start()

	for ws in partition(list(words), num_consumers):
		tasks.put(Task(documents, ws))
	print("--- %s seconds ---" % (time.time() - start_time))
	# num_jobs = 10
	# for i in xrange(num_jobs):
	#     tasks.put(Task(i, i))

	# Add a poison pill for each consumer
	for i in range(num_consumers):
	    tasks.put(None)

	# Wait for all of the tasks to finish
	print("--- %s seconds ---" % (time.time() - start_time))
	tasks.join()

	# Start printing results
	print(len(word_features))
	print("--- %s seconds ---" % (time.time() - start_time))
