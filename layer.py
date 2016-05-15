from neuron import * 

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









