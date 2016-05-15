
import random
from enum import Enum
import numpy as np
from numpy import linalg as LA
import itertools
import math

from layer import *

class LearningType(Enum):
		PERCEPTRON =0
		ADALINE =1


class Trainer:

	def __init__(self):
		
		pass	


	def setMaxEpochs(self,val):
		self.maxEpochs = val

	def getMaxEpochs(self):
		return self.maxEpochs

	def setTrainSet(self,trainSet):
		self.trainSet= trainSet

	def getTrainSet(self):
		return self.trainSet

	def setTargetError(self,error):
		self.targetError=error
	def getCurrentError(self):
		return self.error


	

	def setTrainType(self,tType):

		if tType =="PERCEPTRON":
			self.trainType = LearningType.PERCEPTRON
		elif tType == "ADALINE":
			self.trainType = LearningType.ADALINE

	def setLearningRate(self,v):
		self.learningRate=v


	






	def genLearnedNeuron(self,neuron):

		newNeuron = Neuron(neuron.getNumberOfIn(),neuron.getActiveFunction())
		listW=neuron.getListOfWeightIn()
		listOfIn= neuron.getListOfInput()
		
		if self.trainType == LearningType.PERCEPTRON:
			print "fector: "+str(self.learningRate*self.error*np.array(listOfIn))
			listW=list((self.learningRate*self.error*np.array(listOfIn))+np.array(listW))
			
		elif self.trainType == LearningType.ADALINE:
			assert True,"ADALINE"
		else:
			assert True,"trainType"
		newNeuron.setListOfWeightIn(listW)
		return newNeuron




	def genLearnedLayer(self,layer):


		newLayer = Layer()
		listOfNeurons=layer.getListOfNeurons()
		for n in listOfNeurons:
			
			newLayer.appendNeuron(self.genLearnedNeuron(n))
			


		return newLayer

		




	def train(self,neuralNet):

		epochs=0
		while epochs < self.getMaxEpochs():

			setX= self.getTrainSet().getSetX()
			setY= self.getTrainSet().getSetY()
			result=0
			indexY=0
			for vectorX in setX:

				print "===========================\n\n"

				print "input: "+str(vectorX)
				neuralNet.input(vectorX)
				result=neuralNet.compute()
				

				


				#error Issue 1
				#self.error=  LA.norm( (np.array(result)-np.array(setY[indexY])) )
				self.error = (np.array(setY[indexY])-np.array(result)).sum()

				print "epochs: "+str(epochs)+" result:  "+str(result)+" tar: "+str(setY[indexY])+" error: "+str(self.error)
				print "=====\n\n"

				if pow (self.getCurrentError(),2) >= pow( self.targetError,2):
					print "train!!"

					#input layer
					neuralNet.setInputLayer(self.genLearnedLayer(neuralNet.getInputLayer()))

					#output layer
					neuralNet.setOutputlayer(self.genLearnedLayer(neuralNet.getOutputlayer()))

					hiddenColl = neuralNet.getHiddenlayerCollection()

					#hidden layer
					newHiddenColl = HiddenLayerCollection()
					for layer in hiddenColl:	
						newHiddenColl.append(self.genLearnedLayer(layer))

					neuralNet.setHiddenlayerCollection(newHiddenColl)

						

				indexY+=1

			epochs+=1







class TrainSet:

	def __init__(self):
		self.tmpError=0.0
		pass

	def setX(self,x):
		self.x=x
	def setY(self,y):
		self.y=y

	def getSetX(self):
		return self.x
	def getSetY(self):
		return self.y





