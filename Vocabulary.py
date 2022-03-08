import os
import sys
import json
import numpy as np
import pickle
import Config

class Vocabulary:
	def __init__(self):
		self.config = Config.config

		category = list(pickle.load(open(self.config.category_file, "rb"))) #list:string
		featCate = list(pickle.load(open(self.config.feat_key_file, "rb")))	#list:string
		featVal = list(pickle.load(open(self.config.feat_val_file, "rb")))	#list:string
		cateFK2val = pickle.load(open(self.config.cateFK2val_file, "rb"))	#dic:(key:string 
																			# 	value:dic(key:string value:list_string))
		# 字符串字典
		self.cateFK2val = cateFK2val
		# 大类的id 字符串字典
		self.id2category = category
		self.category2id = dict(zip(self.id2category, range(len(self.id2category))))
		# key-value对的key的id 字符串字典
		self.id2featCate = ["<MARKER>", "<SENT>"] + featCate
		self.featCate2id = dict(zip(self.id2featCate, range(len(self.id2featCate))))
		# 描述类的字典，上面的只有补充符不一样
		self.id2type = ["<GENERAL>"] + featCate
		self.type2id =dict(zip(self.id2type, range(len(self.id2type))))
		# key-value对的value的id 字符串字典
		self.id2featVal = ["<S>", "<ADJ>"] + featVal
		self.featVal2id = dict(zip(self.id2featVal, range(len(self.id2featVal))))
		# 下面两个长度一致，大部分为0,开辟空间用于存储
		self.id2word = ["<S>", "</S>", 0] + [0] * len(featVal)
		self.id2vec = [0] * (3 + len(featVal))
		nxt = 3
		# 词转换为词向量的文件
		# 因为没有提供所以只能推测：
		# 一行为： 单词 词向量
		# 接下来整体表达的是，将未知符和key-value中的value的词向量存在前面（数组格式，查找快一些）
		# 将其它词存在后面
		with open(self.config.wordvec_file, "r") as file:
			for _ in range(self.config.skip_cnt):
				file.readline()
			for line in file:
				line = line.split(" ")
				word = line[0]
				vec = [eval(i) for i in line[1:]]
				if word in featVal:
					self.id2word[nxt] = word
					self.id2vec[nxt] = vec
					nxt += 1
				elif word == "<UNK>":
					self.id2word[2] = "<UNK>"
					self.id2vec[2] = vec
				else:
					self.id2word.append(word)
					self.id2vec.append(vec)
		# 查找是否存在不在wordvec中的value属性，存在则赋值
		for val in featVal:
			if val not in self.id2word: 
				self.id2word.append(val)  #此处应该有问题  应为self.id2word[nxt]=val
				self.id2vec[nxt] = list(np.random.uniform(low=-0.1, high=0.1, size=(self.config.word_dim, )))
				nxt += 1
		assert nxt == len(featVal) + 3	#没有新的关键词  即featval中的词
		# 关键词的数量 
		self.keywords_cnt = nxt
		# 添加未知符到词表内，并随机生成词向量
		fcnt = 2
		if "<UNK>" not in self.id2word:
			self.id2word[2] = "<UNK>"
			fcnt += 1
		for i in range(fcnt):
			self.id2vec[i] = list(np.random.uniform(low=-0.1, high=0.1, size=(self.config.word_dim, )))
		# 对应生成字典
		self.word2id = dict(zip(self.id2word, range(len(self.id2word))))
		# table 包含4个根据词寻找id的字典
		self.table = [self.featCate2id, self.featVal2id, self.word2id, self.type2id]

		self.start_token = 0
		self.end_token = 1

	def lookup(self, word, tpe):
		"""
		:param word:
		:param tpe: 0 for featCate   特征 key
					1 for featVal	特征 value
					2 for word		是否有这个词
					3 for type		特征 cate
		:return:
		"""
		if tpe == 2:
			return self.table[tpe].get(word, self.table[tpe]["<UNK>"])
		else:
			return self.table[tpe][word]
voca = Vocabulary()

