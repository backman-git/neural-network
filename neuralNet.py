import random
from enum import Enum
import numpy as np
from numpy import linalg as LA
import itertools
import math

from layer import *




class NeuralNet:
	


	def __init__(self):
		
		self.inputVector=[]
		self.outputVector=[]
		self.numberOfNeuronOfInputLayer=0
		self.numberOfNeuronOfHiddenLayer=0
		self.numberOfNeuronOfOutputLayer=0


	def setInputLayer(self,layer):
		self.inputLayer=layer

	def getInputLayer(self):
		return self.inputLayer

	def setOutputlayer(self,layer):
		self.outputLayer=layer
	def getOutputlayer(self):
		return self.outputLayer



	def setHiddenlayerCollection(self,layers):
		self.hiddenLayerCollection=layers


	def getHiddenlayerCollection(self):
		return self.hiddenLayerCollection


	def input(self,vector):
		self.inputVector=vector

	def compute(self):

#inputLayer
		
		inputListNeuron=self.inputLayer.getListOfNeurons()

		vector=[]
		idx=0
		for n in inputListNeuron:
			n.setListOfInput(self.inputVector[idx])
			idx+=1
			vector.append(n.getListOfWeightIn())


		vector= list(np.array(vector).flatten())
		inputLayerRes=list(np.array(self.inputVector)*np.array(vector))


	
		#print "inputlayer Weight: " + str(vector)
		#print "inputlayer result: "+str(inputLayerRes)


# hidden layer
		
		hiddenComputeV=inputLayerRes[:]
		for obj in self.hiddenLayerCollection :
			neurons=obj.getListOfNeurons()
			computeResV=[]
			#print "hidden layer"
			for n in neurons:
				#print "neuron weight: "+str(n.getListOfWeightIn())
				n.setListOfInput(hiddenComputeV)
				#netValue  may put to class Neuron
				neuronRes=( np.array(hiddenComputeV)*np.array(n.getListOfWeightIn())).sum()
				#activeFunction
				aFunc=n.getActiveFunction()
				
				neuronRes = aFunc(neuronRes)
				computeResV.append(neuronRes)


			hiddenComputeV=computeResV[:]

		#print "hidden layer result: " + str(hiddenComputeV)

#output layer
		neurons=self.outputLayer.getListOfNeurons()
		computeResV=[]
		for n in neurons:


			n.setListOfInput(hiddenComputeV)
			#netValue
			neuronRes=( np.array(hiddenComputeV)*np.array(n.getListOfWeightIn())).sum()
			#activeFunction
			aFunc=n.getActiveFunction()
			neuronRes = aFunc(neuronRes)
			computeResV.append(neuronRes)



		#print "outLayer result: "+ str(computeResV)


		return computeResV

		


		
		



		
		
			

	def pt(self):

		print "===neuralNet===\n"

		self.inputLayer.pt()

		self.hiddenLayerCollection.pt()

		self.outputLayer.pt()

		print "===neuralNet==="
			






