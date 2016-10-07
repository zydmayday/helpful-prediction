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

	def get_all(self):
		conn = self.conn
		sql = '''SELECT * FROM review;'''
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
	def __init__(self):
		self.client = MongoClient()
		self.db = client.amazon_db


if __name__ == '__main__':
	adb = AmazonDB()
	# adb.create_review_table()
	adb.create_dataset_1_table()
	# adb.insert()
