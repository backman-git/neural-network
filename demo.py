'''
Author: Backman
email: backman.only@gmail.com

Issue:
	1. Don't know how to define the direction/sign of error (vector)
	2. Neuron compute logic should be inside Neuron class
	3. Neuron should not auto gen weight every time
	4. should build a neuron factory





'''
import random
from enum import Enum
import numpy as np
from numpy import linalg as LA
import itertools
import math


from neuron import *
from layer import *
from neuralNet import *
from trainer import *





class LearningType(Enum):
		PERCEPTRON =0
		ADALINE =1


def linearFunction(v):
	return v

def stepFunction(v):
	if v>0:
		return 1
	else :
		return 0




if __name__ =='__main__':

	layerFactory = LayerFactory()
	nn=NeuralNet()
	nn.setInputLayer(layerFactory.type("input").neurons(3).activeFunc(linearFunction).tips(1).getLayer())
	hidden_collection = HiddenLayerCollection()
	hidden_collection.append(layerFactory.type("hidden").neurons(1).activeFunc(linearFunction).tips(3).getLayer())
	nn.setOutputlayer(layerFactory.type("output").neurons(1).activeFunc(stepFunction).tips(1).getLayer())	
	nn.setHiddenlayerCollection(hidden_collection)
	

	
	trainer = Trainer()
	trainSet = TrainSet()
	trainSet.setX([[1.0,0.0,0.0],[1.0,0.0,1.0],[1.0,1.0,0.0],[1.0,1.0,1.0]])
	trainSet.setY([[0.0],[0.0],[0.0],[1.0]])
	trainer.setTrainSet(trainSet)
	trainer.setMaxEpochs(30)
	trainer.setTargetError(0)
	trainer.setTrainType("PERCEPTRON")
	trainer.setLearningRate(0.5)
	trainer.train(nn)

	print "======================"
	print "after training process"
	print "======================"


	nn.input([1.0,0.0,0.0])
	print "0.0 0.0 0.0: "+str(nn.compute())

	nn.input([1.0,0.0,1.0])
	print "1.0 0.0 0.0: "+str(nn.compute())

	nn.input([1.0,1.0,0.0])
	print "1.0 1.0 0.0: "+str(nn.compute())

	nn.input([1.0,1.0,1.0])
	print "1.0 1.0 1.0: "+str(nn.compute())





	



 