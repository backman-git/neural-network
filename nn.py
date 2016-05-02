import random
from enum import Enum
import numpy as np
import itertools


class Neuron:

	def __init__(self,numberOfIn):

		self.listOfWeightIn=[]
		
		for i in range(numberOfIn):
			#r=random.random()
			r=1
			self.listOfWeightIn.append(r)
		

	def getListOfWeightIn(self):

		return self.listOfWeightIn

	def setListOfWeightIn(self,listOfWeightIn):
		self.listOfWeightIn= listOfWeightIn;


	def pt(self):

		print "Input Weights: " +str(self.listOfWeightIn)
		#print "output Weights: "+str(self.listOfWeightOut)



class Layer:
	

	def __init__(self):

		self.listOfNeurons=[]
		self.numberOfNeuronsInLayer=0


	def getListOfNeurons(self):

		return self.listOfNeurons


	def setListOfNeurons(self,neurons):
		self.listOfNeurons = neurons



	def appendNeuron(self,neuron):
		self.numberOfNeuronsInLayer+=1
		self.listOfNeurons.append(neuron)


	def getNumberOfNeuronsInLayer(self):
		return self.numberOfNeuronsInLayer

	def setNumberOfNeuronsInLayer(numberOfNeuronsInLayer):
		self.numberOfNeuronsInLayer=numberOfNeuronsInLayer


	def pt(self):


		n=1
		for neuron in self.listOfNeurons:
			print "Neuron #"+str(n)
			n+=1
			neuron.pt()
		print "=======\n"






class InputLayer(Layer):

	def __init__(self):
		Layer.__init__(self)

	def pt(self):
		print "+++inputLayer+++"
		Layer.pt(self)


class HiddenLayer(Layer):

	def __init__(self):
		Layer.__init__(self)

	def pt(self):
	
		Layer.pt(self)


class OutputLayer(Layer):
	
	def __init__(self):
		Layer.__init__(self)

	def pt(self):
		print "+++OutputLayer+++"
		Layer.pt(self)


class LayerFactory:

	def __init__(self):
		self.nTipOfNeuron=0
		self.nNeuron=0
		self.typeOfLayer=""
	def type(self,t):
		self.typeOfLayer=t
		return self

	def tips(self,n):
		self.nTipOfNeuron=n
		return self

	def neurons(self,n):
		self.nNeuron=n
		return self

	def getLayer(self):

		#build neuronin_layer=InputLayer()
		listOfNeurons=[]
		for i in range(self.nNeuron):
			listOfNeurons.append(Neuron(self.nTipOfNeuron))
		
		if self.typeOfLayer == "input":
			iLayer =  InputLayer()
			iLayer.setListOfNeurons(listOfNeurons)
			return iLayer

		elif self.typeOfLayer =="output":
			oLayer = OutputLayer()
			oLayer.setListOfNeurons(listOfNeurons)
			return oLayer
		elif self.typeOfLayer =="hidden":
			hLayer = HiddenLayer()
			hLayer.setListOfNeurons(listOfNeurons)
			return hLayer
		else :
			return None






class HiddenLayerCollection(list):


	def pt(self):
		print ("+++ HIDDEN LAYERS+++\n")

		n=1
		for layer in self:
			print "Hidden Layer #"+str(n)
			n+=1
			layer.pt()















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
	def setHiddenlayerCollection(self,layers):
		self.hiddenLayerCollection=layers


	def input(self,vector):
		self.inputVector=vector

	def compute(self):

#inputLayer
		
		inputListNeuron=self.inputLayer.getListOfNeurons()
		vector=[]
		for n in inputListNeuron:
			vector.append(n.getListOfWeightIn())
		vector= list(np.array(vector).flatten())
		inputLayerRes=list(np.array(self.inputVector)*np.array(vector))


		print "\ninputvector"
		print self.inputVector
		print "\ninputlayer"
		print vector
		print "\n==inputlayer result=="
		print inputLayerRes


# hidden layer
		
		hiddenComputeV=inputLayerRes[:]
		for obj in self.hiddenLayerCollection:
			neurons=obj.getListOfNeurons()
			computeResV=[]
			for n in neurons:
				neuronRes=( np.array(hiddenComputeV)*np.array(n.getListOfWeightIn())).sum()
				computeResV.append(neuronRes)
			hiddenComputeV=computeResV[:]

		print "\n=== hidden layer result"
		print hiddenComputeV

#output layer
		neurons=self.outputLayer.getListOfNeurons()
		computeResV=[]
		for n in neurons:
			neuronRes=( np.array(hiddenComputeV)*np.array(n.getListOfWeightIn())).sum()
			computeResV.append(neuronRes)
		print "\n=== outLayer result==="
		print computeResV

		


		
		



		
		
			

	def pt(self):

		print "===neuralNet===\n"

		self.inputLayer.pt()

		self.hiddenLayerCollection.pt()

		self.outputLayer.pt()

		print "===neuralNet==="
			








class TrainSet:

	def __init__(self):
		self.x=[]
		self.y=[]

	def set_X(self,obj):
		self.x.append(x)
	def set_Y(self,obj):
		self.y.append(y)

	def get_setX(self):
		return self.x
	def get_setY(self):
		return self.y







class Training:


	def getEpochs(self):
		return self.epochs

	def setMaxEpochs(self,val):
		self.maxEpochs = val

	def getMaxEpochs(self):
		return self.maxEpochs

	def setTrainSet(self,trainSet):
		self.trainSet= trainSet

	def getTrainSet(self):
		return self.trainSet



	class LearningType(Enum):
		PERCEPTRON =0
		ADALINE =1



	def train(self,neuralNet):


		while self.getEpochs() < self.getMaxEpochs():


			setX= self.getTrainSet().get_setX

			for vectorX in setX:
				neuralNet.setInput(vectorX)
				neuralNet.compute()
			
			







if __name__ =='__main__':

	layerFactory = LayerFactory()
	nn=NeuralNet()

	nn.setInputLayer(layerFactory.type("input").neurons(4).tips(1).getLayer())
	nn.setOutputlayer(layerFactory.type("output").neurons(1).tips(2).getLayer())

	hidden_collection = HiddenLayerCollection()
	hidden_collection.append(layerFactory.type("hidden").neurons(3).tips(4).getLayer())
	hidden_collection.append(layerFactory.type("hidden").neurons(2).tips(3).getLayer())


	nn.setHiddenlayerCollection(hidden_collection)
	nn.pt()
	nn.input([1,2,3,4])
	nn.compute()


	



 