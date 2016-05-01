import random
from enum import Enum
import numpy as np
import itertools


class Neuron:

	def __init__(self,numberOfIn):

		self.listOfWeightIn=[]
		
		for i in range(numberOfIn):
			#r=random.random()
			r=2
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


	def setListOfNeurons(self):
		self.listOfNeurons = listOfNeurons


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
		self.type=""
		self.nTipOfNeuron=0
		self.nNeuron=0

	def type(self,t):
		self.type=t
		return self

	def tips(self,n):
		self.nTipOfNeuron=n
		return self

	def neurons(self,n):
		self.nNeuron=n
		return self

	def getLayer(self):

		if self.type == "input":
			return 

		elif self.type =="output":

		else self.type =="hidden"






class HiddenLayerCollection(list):


	def pt(self):
		print ("+++ HIDDEN LAYERS+++\n")

		n=1
		for layer in self:
			print "Hidden Layer #"+str(n)
			n+=1
			layer.pt()



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


		print "inputvector"
		print self.inputVector
		print "inputlayer"
		print vector
		print "==inputlayer result=="
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
		print "=== hidden layer result"
		print hiddenComputeV

#output layer
		neurons=self.outputLayer.getListOfNeurons()
		computeResV=[]
		for n in neurons:
			neuronRes=( np.array(hiddenComputeV)*np.array(n.getListOfWeightIn())).sum()
			computeResV.append(neuronRes)
		print "=== outLayer result==="
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

	nn = NeuralNet() 

	

	


	in_layer=InputLayer()
	for i in range(4):
		n = Neuron(1)
		in_layer.appendNeuron(n)

	in_layer.pt()

	out_layer=OutputLayer()
	for i in range(1):
		n = Neuron(1)
		out_layer.appendNeuron(n)
	out_layer.pt()

	

	hidden_collection=HiddenLayerCollection()
	
	hidden_layer = HiddenLayer()
	for j in range(5):
		n = Neuron(4)
		hidden_layer.appendNeuron(n)

	hidden_collection.append(hidden_layer)

	hidden_layer2 = HiddenLayer()

	for j in range(2):
		n = Neuron(5)
		hidden_layer2.appendNeuron(n)

	hidden_collection.append(hidden_layer2)





		

	hidden_collection.pt()

	print in_layer.getListOfNeurons()

	nn=NeuralNet()

	nn.setInputLayer(in_layer)
	nn.setOutputlayer(out_layer)
	nn.setHiddenlayerCollection(hidden_collection)
	nn.pt()
	nn.input([1,2,3,4])
	nn.compute()





 