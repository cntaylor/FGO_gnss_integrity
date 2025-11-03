#!/usr/bin/env/ python 

'''
Class to form range and phase observation models
'''
import numpy as np

import Const
import Transform
import math
physConst=Const.Const()
earth= Const.Earth()

def computedRange(satXYZ, staXYZ, satClockBias,staClockBias, tropZ):
	satXYZ=np.array(satXYZ)
	staXYZ=np.array(staXYZ)
	return np.linalg.norm(satXYZ-staXYZ)+ \
		(staClockBias-satClockBias) + \
		dryTrop(satXYZ,staXYZ)+ \
		tropZ*mapWetTrop(satXYZ,staXYZ)


def dryTrop(satXYZ,staXYZ):
	return tropDryDelay(staXYZ)*mapDryTrop(satXYZ,staXYZ)

def mapWetTrop(satXYZ,staXYZ):
	ElAz=Transform.calcElAz(satXYZ,staXYZ)
	map=1/np.sqrt(1.0-(np.cos(ElAz[0])/1.001)**2)
	return map

def mapDryTrop(satXYZ,staXYZ):
	return mapWetTrop(satXYZ,staXYZ)

def tropDryDelay(usrXYZ, humid = 0.5):
	'''
	Saastemoinen Model from Dry Trop
	'''

	# Atmospheric Parameters
	p=1013 #std pressure in mbar
	T=288.15 #temp in Kelvin
	hp=0
	ht=1

	usrllh=Transform.xyz2llh(usrXYZ)
	hu=usrllh[2]/1000 # usr heighgt in km

	tk=T-6.5*hu # just model temp trop of lower atm up to tropopause
	te=T+6.5*(ht-hp)
	em=5.2459587
	psea=p*(tk/te)**em
	Nd0=77.624*psea/tk

	hd=40136+148.72*(tk-288.16)
	delta=((1e-6)/5)*(Nd0*hd)

	return delta

def ionoObliquityFactor(usrXYZ, satXYZ, hI=350000):

	'''
	Obliquity Factor from Misra and Enge 5.26 pp 164
	'''
	aElAz=Tranform.calcElAz(satXYZ,usrXYZ)
	zenithAngle = math.pi/2 - ElAz[0]

	return  math.sqrt( 1.0 - ( (Earth.R * sin(zenithAngle) )/ ( Earth.E + hI )  ) )

