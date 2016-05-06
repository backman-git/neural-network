import random
from enum import Enum
import numpy as np
from numpy import linalg as LA
import itertools
import math






def linearFunction(v):
	return v










class Neuron:

	def __init__(self,numberOfIn):

		self.listOfWeightIn=[]
		self.Afunction=linearFunction
		self.numberOfIn = numberOfIn
		
		for i in range(numberOfIn):
			#r=random.random()
			r=1.0
			self.listOfWeightIn.append(r)
		

	def getNumberOfIn(self):
		return self.numberOfIn


	def getListOfWeightIn(self):

		return self.listOfWeightIn

	def setListOfWeightIn(self,listOfWeightIn):
		self.listOfWeightIn= listOfWeightIn;

	def setActiveFunction(self,func):
		self.Afunction =func

	def getActiveFunction(self):
		return self.Afunction

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


	# active function (all neurons are same ) may need change
	def activeFunc(self,f):
		self.aFunc= f
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
		for obj in self.hiddenLayerCollection :
			neurons=obj.getListOfNeurons()
			computeResV=[]
			for n in neurons:
				#netValue
				neuronRes=( np.array(hiddenComputeV)*np.array(n.getListOfWeightIn())).sum()
				#activeFunction
				aFunc=n.getActiveFunction()
				neuronRes = aFunc(neuronRes)
				computeResV.append(neuronRes)


			hiddenComputeV=computeResV[:]

		print "\n=== hidden layer result"
		print hiddenComputeV

#output layer
		neurons=self.outputLayer.getListOfNeurons()
		computeResV=[]
		for n in neurons:
			#netValue
			neuronRes=( np.array(hiddenComputeV)*np.array(n.getListOfWeightIn())).sum()
			#activeFunction
			aFunc=n.getActiveFunction()
			neuronRes = aFunc(neuronRes)

			computeResV.append(neuronRes)
		print "\n=== outLayer result==="
		print computeResV


		return computeResV

		


		
		



		
		
			

	def pt(self):

		print "===neuralNet===\n"

		self.inputLayer.pt()

		self.hiddenLayerCollection.pt()

		self.outputLayer.pt()

		print "===neuralNet==="
			








class TrainSet:

	def __init__(self):
		pass

	def setX(self,x):
		self.x=x
	def setY(self,y):
		self.y=y

	def getSetX(self):
		return self.x
	def getSetY(self):
		return self.y







class Trainer:

	def __init__(self):
		
		self.error=0


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


	class LearningType(Enum):
		PERCEPTRON =0
		ADALINE =1

	def setTrainType(self,tType):

		if tType =="PERCEPTRON":
			self.trainType = self.LearningType.PERCEPTRON
		elif tType == "ADALINE":
			self.trainType = self.LearningType.ADALINE




	


	def calWeight(self,weight):

		return weight





	def teachLayer(self,layer):

		listOfNeurons=layer.getListOfNeurons()

		for n in listOfNeurons:
			newNeuron= Neuron(n.getNumberOfIn)
			listOfWeights=[]
			for w in n.getListOfWeightIn():
				listOfWeights.append( self.calWeight(w))

			newNeuron.setListOfWeightIn(listOfWeights)
			newLayer= 



		return layer

		




	def train(self,neuralNet):

		epochs=0
		while epochs < self.getMaxEpochs():

			setX= self.getTrainSet().getSetX()
			setY= self.getTrainSet().getSetY()
			result=0
			indexY=0
			for vectorX in setX:

				print vectorX
				neuralNet.input(vectorX)
				result=neuralNet.compute()
				

				print "epochs: "+str(epochs)+" result:  "+str(result)
				#error
				
				self.error=  LA.norm( (np.array(result)-np.array(setY[indexY])) )
				print self.error
				if self.getCurrentError() > self.targetError:

					#input layer
					neuralNet.setInputLayer(self.teachLayer(neuralNet.getInputLayer()))
					#output layer
					neuralNet.setOutputlayer(self.teachLayer(neuralNet.getOutputlayer()))

					hiddenColl = neuralNet.getHiddenlayerCollection()

					#hidden layer
					newHiddenColl = HiddenLayerCollection()
					for layer in hiddenColl:	
						newHiddenColl.append(self.teachLayer(layer))

					neuralNet.setHiddenlayerCollection(newHiddenColl)




						


				indexY+=1

			epochs+=1





if __name__ =='__main__':

	layerFactory = LayerFactory()
	nn=NeuralNet()

	nn.setInputLayer(layerFactory.type("input").neurons(3).activeFunc(linearFunction).tips(1).getLayer())
	nn.setOutputlayer(layerFactory.type("output").neurons(1).activeFunc(linearFunction).tips(1).getLayer())

	hidden_collection = HiddenLayerCollection()
	hidden_collection.append(layerFactory.type("hidden").neurons(1).activeFunc(linearFunction).tips(3).getLayer())
	hidden_collection.append(layerFactory.type("hidden").neurons(1).activeFunc(linearFunction).tips(1).getLayer())
	
	nn.setHiddenlayerCollection(hidden_collection)
	

	'''
	nn.pt()
	nn.input([1,2,3])
	nn.compute()
	'''
	trainer = Trainer()

	trainSet = TrainSet()
	trainSet.setX([[1.0,0.0,0.0],[1.0,0.0,1.0],[1.0,1.0,0.0],[1.0,1.0,1.0]])
	trainSet.setY([0.0,0.0,0.0,1.0])
	trainer.setTrainSet(trainSet)
	trainer.setMaxEpochs(10)
	trainer.setTargetError(1.0)
	trainer.setTrainType("PERCEPTRON")
	trainer.train(nn)
	



	



 