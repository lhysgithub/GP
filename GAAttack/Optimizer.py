import tensorflow as tf

class GAOptimizer(object):
    def __init__(self,sess,PopulationSize):
        self.PopulationSize=PopulationSize
    def minimize(self,loss,person):

