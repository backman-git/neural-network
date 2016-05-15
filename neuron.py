
import random
from enum import Enum
import numpy as np
from numpy import linalg as LA
import itertools
import math


def linearFunction(v):
	return v


class Neuron:

	def __init__(self,numberOfIn,func):

		self.listOfWeightIn=[]
		self.Afunction=linearFunction
		self.numberOfIn = numberOfIn
		self.Afunction=func

		
		for i in range(numberOfIn):
			r=random.random()
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







