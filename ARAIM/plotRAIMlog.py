#!/usr/bin/env python

import sys, os
import numpy as np
import matplotlib.pyplot as plt


# load the input file
inFile = open( sys.argv[1] ).readlines()

HPL = []
VPL = []
NFaultMax = []
NFaultModesConsidered = []
Time = []
vertAcc95 = []
vertAccFF = []
EMT = []
posSoln = []
nSat = []
nGPS = []
nGAL = []
chiSqStat = []
EastFail = []
NorthFail = []
VerticalFail = []

# put file string keys into dict to avoid a lot of
# code re-write with file changes
fileLineKey = {}
fileLineKey['NFaultMax'] = 'N Fault Max Determined'
fileLineKey['posSoln']= 'Position Solution'
fileLineKey['nSat'] = 'Number of'
fileLineKey['FaultModesConsidered'] = 'Fault Modes Evaluated'
fileLineKey['Time'] = 'Time' # this line also has HPL, VPL and EMT
fileLineKey['vertAcc'] = 'Vertical Accuracy' # this line has 95% and 10e-7 FF
fileLineKey['chiSq'] = 'Chi-Square Stat' # this line also has {East,North,Verical}Fail

time2Sec = 1.0/( 10.0 ) # specific to matlab sim
for line in inFile:
	if line.rfind( fileLineKey['NFaultMax'] ) > -1 :
		line = line.split()
		NFaultMax.append( float( line[6] ) )
	
	elif line.rfind( fileLineKey['posSoln'] ) > -1 :
		line = line.split()
		pos = []
		pos.append( float( line[3] ) )
		pos.append( float( line[4] ) )
		pos.append( float( line[5] ) )
		posSoln.append( pos )
	
	elif line.rfind( fileLineKey['nSat'] ) > -1 :
		line = line.split()
		nGPS.append( float( line[3] ) )
		nGAL.append( float( line[8] ) )
		nSat.append( float( line[3] )+ float( line[8] ) )
	
	elif line.rfind( fileLineKey['FaultModesConsidered'] ) > -1 :
		line = line.split()
		NFaultModesConsidered.append( float( line[3] ) )
	
	elif line.rfind( fileLineKey['Time'] ) > -1 :
		line = line.split()
		# note the commas on some of these feilds!
		Time.append( float( line[1][:-1] )*time2Sec ) # comma
		HPL.append( float( line[3][:-1] ) )  # comma
		VPL.append( float( line[6] ) )
	
	elif line.rfind( fileLineKey['chiSq'] ) > -1  and (line.rfind('print') ==-1) :
		line = line.split()
		chiSqStat.append( float( line[4] ) )
		EastFail.append( float( line[9] ) )
		NorthFail.append( float( line[12] ) )
		VerticalFail.append( float( line[15] ) )
		
	
	elif (line.rfind( fileLineKey['vertAcc'] ) > -1) and (line.rfind('print') ==-1):
		line = line.split()
		print(line[3])
		vertAcc95.append( float( line[3][0:-1] ) ) # comma
		vertAccFF.append( float( line[8][0:-1] ) ) # comma
		EMT.append( float( line[10] ) )

	else:
		continue


keys=['East','North','Vertical']
testStatMats=[]
for i in range(3):
	fileLines = os.popen('grep -v All %s | grep -v Time | grep -v Accuracy |\
			grep %s'%( sys.argv[1], keys[i] ) ).readlines()
	
	nRows = len(fileLines)
	nCols = []
	for line in fileLines:
		nCols.append( len(line.split()) - 1 )
	if not nCols:
		continue
	nCols = max(nCols)
	testStats = np.empty((nRows,nCols))
	testStats.fill(np.nan)
	ind = 0
	for line in fileLines:
		nCol = len(line.split()) - 1 
		noComma=[s.replace(',', '')  for s in line.split()[1:]]
		
		testStats[ind,0:nCol] =  [float(i) for i in noComma]
		ind = ind + 1
	testStatMats.append(testStats)


f, ax = plt.subplots(4, sharex=True)
ax[0].plot(Time,nSat,'k',Time,nGPS,'b',Time,nGAL,'g--', linewidth=2.0)
ax[0].set_title('GNSS Advanced RAIM Baseline Algorithm', fontsize=12)
ax[0].set_ylabel(' # of SVs ') 
ax[0].grid()
ax[0].set_ylim([3, 15])
ax[0].set_xlim([-5, 200])
ax[0].legend(['# SV','# GPS','# Galileo'], fontsize = 10, loc =6)

ax[1].plot(Time,VPL,'r',Time,HPL,'b',Time, vertAcc95,'g',Time, vertAccFF, 'm',Time, EMT, 'c', linewidth=2.0 )
ax[1].grid()
ax[1].set_title('Protection Limits and Accuracy Estimates', fontsize=12)
plt.xlabel('Time (sec)')
ax[1].set_ylabel('Meters')
ax[1].legend(['VPL','HPL','Vacc 95%','Vacc 10e-7','EMT'], fontsize = 10, loc = 6)


ax[2].plot(Time, chiSqStat, 'r')
ax[2].set_ylim([-.1,50])
ax[2].set_title('All in view Failure Stat ', fontsize=12)
ax[2].grid()
ax[2].legend(['All in view Chi^2'], fontsize = 10, loc = 6)
ax[3].plot( Time, EastFail, 'ko',Time, NorthFail,'g+',Time, VerticalFail, 'b*',linewidth=2.0)
ax[3].legend(['# Modes East Fail','# Modes North Fail','# Modes Vertical Fail'],fontsize = 10, loc = 6)
ax[3].set_title('Component Failure Modes', fontsize=12)
ax[3].grid()
plt.savefig(sys.argv[1].split('.')[0]+'.png')

#fTestStat, axTestStat = plt.subplots(3, sharex=True)
#threshY=[1.0,1.0]
#threshX=[0, 2000]
#for i in range(3):
#	axTestStat[i].plot(threshX,threshY,'k-')
#	print (len(Time))
#	print( testStatMats)
#	axTestStat[i].plot(Time,testStatMats[i])
#	axTestStat[i].grid()
#	axTestStat[i].set_ylabel(keys[i])
#	axTestStat[i].set_ylim([0 ,4.0])

#plt.xlabel('Time (s)')
#axTestStat[0].set_title(' Multi-GNSS ARAIM Solution Mode Test Statistics ')

#plt.savefig(sys.argv[1].split('.')[0]+'TestStats'+'.png')
