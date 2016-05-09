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







class Neuron:

	def __init__(self,numberOfIn,func):

		self.listOfWeightIn=[]
		self.Afunction=linearFunction
		self.numberOfIn = numberOfIn
		self.Afunction=func

		
		for i in range(numberOfIn):
			#r=random.random()
			r=1.0
			self.listOfWeightIn.append(r)
		

	
	
		



	def setListOfInput(self,inputs):
		self.ListOfInput=inputs

	def getListOfInput(self):
		return self.ListOfInput

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
			listOfNeurons.append(Neuron(self.nTipOfNeuron,self.aFunc))
		
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
		idx=0
		for n in inputListNeuron:
			n.setListOfInput(self.inputVector[idx])
			idx+=1
			vector.append(n.getListOfWeightIn())


		vector= list(np.array(vector).flatten())
		inputLayerRes=list(np.array(self.inputVector)*np.array(vector))


	
		print "inputlayer Weight: " + str(vector)
		print "inputlayer result: "+str(inputLayerRes)


# hidden layer
		
		hiddenComputeV=inputLayerRes[:]
		for obj in self.hiddenLayerCollection :
			neurons=obj.getListOfNeurons()
			computeResV=[]
			print "hidden layer"
			for n in neurons:
				print "neuron weight: "+str(n.getListOfWeightIn())
				n.setListOfInput(hiddenComputeV)
				#netValue  may put to class Neuron
				neuronRes=( np.array(hiddenComputeV)*np.array(n.getListOfWeightIn())).sum()
				#activeFunction
				aFunc=n.getActiveFunction()
				
				neuronRes = aFunc(neuronRes)
				computeResV.append(neuronRes)


			hiddenComputeV=computeResV[:]

		print "hidden layer result: " + str(hiddenComputeV)

#output layer
		neurons=self.outputLayer.getListOfNeurons()
		computeResV=[]
		for n in neurons:


			n.setListOfInput(hiddenComputeV)
			#netValue
			neuronRes=( np.array(hiddenComputeV)*np.array(n.getListOfWeightIn())).sum()
			#activeFunction
			aFunc=n.getActiveFunction()
			print "------------"+str(neuronRes)
			neuronRes = aFunc(neuronRes)
			print "------------"+str(neuronRes)
			computeResV.append(neuronRes)



		print "outLayer result: "+ str(computeResV)


		return computeResV

		


		
		



		
		
			

	def pt(self):

		print "===neuralNet===\n"

		self.inputLayer.pt()

		self.hiddenLayerCollection.pt()

		self.outputLayer.pt()

		print "===neuralNet==="
			








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

		newNeuron = Neuron(neuron.getNumberOfIn(),neuron.getActiveFunction)
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
				if  pow (self.getCurrentError(),2) >=pow(self.targetError,2):

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





if __name__ =='__main__':

	layerFactory = LayerFactory()
	nn=NeuralNet()

	nn.setInputLayer(layerFactory.type("input").neurons(3).activeFunc(stepFunction).tips(1).getLayer())

	hidden_collection = HiddenLayerCollection()
	hidden_collection.append(layerFactory.type("hidden").neurons(1).activeFunc(linearFunction).tips(3).getLayer())

	nn.setOutputlayer(layerFactory.type("output").neurons(1).activeFunc(stepFunction).tips(1).getLayer())

	
	nn.setHiddenlayerCollection(hidden_collection)
	

	'''
	nn.pt()
	nn.input([1,2,3])
	nn.compute()
	'''
	trainer = Trainer()

	trainSet = TrainSet()
	trainSet.setX([[1.0,0.0,0.0],[1.0,0.0,1.0],[1.0,1.0,0.0],[1.0,1.0,1.0]])
	trainSet.setY([[0.0],[0.0],[0.0],[1.0]])
	trainer.setTrainSet(trainSet)
	trainer.setMaxEpochs(10)
	trainer.setTargetError(0.002)
	trainer.setTrainType("PERCEPTRON")
	trainer.setLearningRate(0.2)
	trainer.train(nn)
	



	



 