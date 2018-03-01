import sys
import tensorflow as tf
import numpy as np

BinarySerachSteps = 9 	# 二分法寻找超参次数
MaxIterations = 10000 	# 最大迭代次数
AbortEarly = True		# 是否提前退出
Targeted = True 		# 是否针对性错分
Confidence = 0			# 对抗样本置信度
InitialConsts = 1e-3	# 初始超参数

class GA:
	def __init__(self,sess,model,batch_size,Confidence=Confidence,Targeted=Targeted,
		BinarySerachSteps=BinarySerachSteps,MaxIterations=MaxIterations,AbortEarly=AbortEarly,
		InitialConsts=InitialConsts,boxmin=-0.5,boxmax=0.5):
	ImageSize,NumChannels,NumLabels = model.image_size, model.num_channels, model.num_labels

	#self.GAGraph = tf.Graph()
	self.GlobalSess = sess
	self.Targeted = Targeted
	self.MaxIterations = MaxIterations
	self.BinarySerachSteps = BinarySerachSteps
	self.AbortEarly = AbortEarly
	self.Confidence = Confidence 
	self.InitialConsts = InitialConsts
	self.BatchSize = batch_size
	#self.Repeat = BinarySerachSteps >= 10
	self.ChromosomesNumber = 100

	self.ImageSize = ImageSize
	self.NumChannels = NumChannels
	self.ImageShape = (self.ChromosomesNumber,self.BatchSize,ImageSize,ImageSize,NumChannels)
	self.LabShape = (self.ChromosomesNumber,self.BatchSize,NumLabels)
	
	self.Modifier = tf.Variable(np.zeros(self.BatchSize,ImageSize,ImageSize,NumChannels),dtype = np.float32)
	
	self.Chromosomes = tf.Variable(np.random.random(self.ImageShape),dtype = tf.float32)
	# 是否存在每次需要重新初始化PC，PM的问题
	self.ChromosomesPC = tf.Variable(np.random.random(self.ImageShape),dtype = tf.float32)
	self.ChromosomesPM = tf.Variable(np.random.random(self.ImageShape),dtype = tf.float32)
		
	self.Imgs = tf.Variable(np.zeros(self.ImageShape),dtype = tf.float32)
	self.Labs = tf.Variable(np.zeros(self.LabShape),dtype = tf.float32)
	self.Consts = tf.Variable(np.zeros(self.ChromosomesNumber,self.BatchSize),dtype = tf.float32)
	
	self.AssignImgs = tf.placeholder(tf.float32,self.ImageShape)
	self.AssignLabs = tf.placeholder(tf.float32,self.LabShape)
	self.AssignConsts = tf.placeholder(tf.float32,(self.ChromosomesNumber,self,BatchSize))

	self.NewImage = Imgs + Chromosomes
	
	# 是否存在不能预测的问题
	self.OutPut = model.predict(self.NewImage)
	self.L2Distance = tf.reduce_sum(tf.square(self.Chromosomes),[2,3,4])
	Real = tf.reduce_sum((self.Labs)*self.OutPut,2)
	Other = tf.reduce_max((1-self.Labs)*OutPut,2)
	
	if self.Targeted :
		loss1 = tf.maximun(0.0,Other - Real + self.Confidence)
	else :
		loss1 = tf.maximun(0.0,Real - Other + self.Confidence)

	self.loss1 = tf.reduce_sum(self.Consts*loss1,1)
	self.loss2 = tf.tf.reduce_sum(self.L2Distance,1)
	self.Fitness = self.loss1 + self.loss2

	self.setup = []
	self.setup.apppend(self.ImgsTensor.assign(self.AssignImgs))
	self.setup.apppend(self.ImgsTensor.assign(self.AssignImgs))
	self.setup.apppend(self.ImgsTensor.assign(self.AssignImgs))
	
	def Attack(self,Imgs,Targets):
		r = []
		for i in range(0,len(Imgs),self.BatchSize):
			print("tick",i)
			r.extend(self.AttackBatch(Imgs[i:i+self.BatchSize],Targets[i:i+self.BatchSize]))
		return np.array(r)
	def AttackBatch(self,Imgs,Labs):

		#BatchSize = self.BatchSize
		# with self.GAGraph.as_default():
		#ImageShape = (self.BatchSize,ImageSize,ImageSize,NumChannels)
		
		# 开始迭代进化
		for i in range(self.MaxIterations):
					
		





		# 只清理不删除
		self.GAGraph.clear()



