#!/usr/bin/env python 

''' 
Class to store physical constants
'''

import numpy as np

class  Const:
	def __init__(self):
		self.speedOfLight= 299792458 # m/s
		self.gravity = 9.80665 #m/s^2
		self.deg2rad = 0.0175 
		self.ft2m = 0.3048
		self.f1 = 1575.42e6 # Hz
		self.f2 = 1227.60e6 # Hz
		self.f5 = 1176.45e6 # Hz
		self.gps2unix= 315964800 # s

class Earth:
	def __init__(self):
		self.a = 6378137.0000 # semi-major
		self.b = 6356752.3142 # semi-minor
		self.f = 1/298.257223560
		self.t = ((1-self.f)*(1-self.f))
		self.e = np.sqrt(1-(self.b/self.a)**2)
		self.bSqr= self.b*self.b
		self.eSqr= self.e*self.e
		self.ep=self.e*(self.a/self.b)
		self.ESqr=self.a**2-self.b**2
		self.RotationRate = 7.292115e-5 # rad/sec
		self.R = self.a
	

	
