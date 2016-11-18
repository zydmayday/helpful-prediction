'''
Try to get data from database,
provide the interface of how to get data from db
'''
import sqlite3
import codecs

from pymongo import MongoClient

class AmazonDB:

	def __init__(self, dbname='amazon.db'):
		self.conn = sqlite3.connect(dbname)

	def create_review_table(self):
		conn = self.conn
		sql = """DROP TABLE IF EXISTS REVIEW;"""
		conn.execute(sql)
		sql = """CREATE TABLE REVIEW
		       (ID INT PRIMARY KEY     NOT NULL,
		       	PRODUCTID VARCHAR(20)	NOT NULL,
		       REVIEWDATE        DATE     NOT NULL,
		       RATING        INT     NOT NULL,
		       HELPFUL        INT     NOT NULL,
		       TITLE           TEXT    NOT NULL,
		       CONTENT        TEXT     NOT NULL);"""
		conn.execute(sql)

	def create_dataset_1_table(self):
		conn = self.conn
		sql = """DROP TABLE IF EXISTS DATASET_1;"""
		conn.execute(sql)
		sql = """CREATE TABLE DATASET_1 
		       (ID INTEGER PRIMARY KEY     NOT NULL,
		       	X TEXT	NOT NULL,
		        Y TEXT  NOT NULL);"""
		conn.execute(sql)

	def insert(self, file_name='/home/data/amazon/MProductReviewsLatest.txt'):
		conn = self.conn
		reviews = codecs.open(file_name, encoding='utf-8', errors='ignore')
		reviews.readline()
		template = '''INSERT INTO REVIEW (ID,PRODUCTID,REVIEWDATE,RATING,HELPFUL,TITLE,CONTENT) VALUES (%s,"%s",%s,%s,%s,"%s","%s")'''

		for line in reviews.readlines():
			try:
				l = line.split('\t')
				l[10] = l[10].replace("\"", "\'")
				l[11] = l[11].replace("\"", "\'")
				sql = template % (l[0],l[5],l[6],l[7],l[8],l[10],l[11])
				conn.execute(sql)
			except:
				print(line)
				pass
		conn.commit()

	def get_all(self, cols='*', gt=0):
		conn = self.conn
		if type(cols) == list:
			cols = ','.join(cols) 
		sql = '''SELECT %s FROM review;''' % cols
		if gt:
			sql = '''SELECT %s from review where productid in (select productid from review group by productid having count(id) >= %d);''' % (cols, gt)
		cur = conn.execute(sql)
		return cur

	def get_content_helpful(self, cols="content, helpful", gt=0, lt=0):
		conn = self.conn
		if type(cols) == list:
			cols = ','.join(cols) 
		if gt:
			sql = '''SELECT %s from review where helpful>= %d;''' % (cols, gt)
		if lt:
			sql = '''SELECT %s from review where helpful<= %d;''' % (cols, lt)
		cur = conn.execute(sql)
		return cur

	def count_rating(self):
		sql = '''SELECT rating,count(1) from review group by rating;'''
		cur = self.conn.execute(sql)
		return cur

	def execute(self, sql):
		cur = self.conn.execute(sql)
		return cur

	def save(self, sql):
		self.conn.execute(sql)
		self.conn.commit()

	def get_helpful_group_by_product(self):
		sql = '''SELECT productid,sum(helpful) from review group by productid'''
		cur = self.conn.execute(sql)
		return cur

	def get_all_helpful(self):
		sql = '''SELECT helpful, productid from review'''
		return self.conn.execute(sql)

class AmamzonMongoDb:
	def __init__(self, user='testuser_001', psword='testuser_001', dbname='db_001', url='mongodb://10.63.60.15:27017/'):
		self.client = MongoClient()
		self.db = client[dbname]
		self.db.authenticate(uesr, psword)

	def table(self, tbname='test'):
		return self.db[tbname]


if __name__ == '__main__':
	adb = AmazonDB()
	# adb.create_review_table()
	adb.create_dataset_1_table()
	# adb.insert()
