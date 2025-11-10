#!/usr/bin/env python

import os

png=os.getenv('PNG')
prn2gpsFile=png+'/pylib/pyPNGVar/PRN_GPS'
prn2gpsData=open(prn2gpsFile).readlines()

i=0
prn2gpsDict={}
gps2prnDict={}
for line in prn2gpsData:
	i=i+1
	if i>1:
		line=line.split()
		prn2gpsDict[int(line[3])]=int(line[2])
		gps2prnDict[int(line[2])]=int(line[3])



def prn2gps(prn):
	return prn2gpsDict[int(prn)]

def gps2prn(gps):
        return gps2prnDict[int(gps)]


def find(a, func):
	""" 
	implementation of a matlab-style find function for python
	source: http://stackoverflow.com/questions/5957470/matlab-style-find-function-in-python

	"""

	return [i for (i, val) in enumerate(a) if func(val)]


def null(a, rtol=1e-5):
	"""
	find the null space of a matrix, given the svd
	source: http://www.widecodes.com/0xSVejjggg/a-simple-matlablike-way-of-finding-the-null-space-of-a-small-matrix-in-numpy-and-number-formatting-duplicate.html
	Inputs:
		small matrix, numerical tolerance
	Output:
		rank, null space
	"""
	u, s, v = np.linalg.svd(a)
	rank = (s > rtol*s[0]).sum()
	return rank, v[rank:].T.copy()

