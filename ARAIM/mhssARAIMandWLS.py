#!/usr/bin/env python

import sys
import IntegrityRAIMandWLS2 as Integrity
import MatlabSim2PythonSim
import numpy as np
import random
global i 
i = 0


simDir = sys.argv[1]

# load up data
data = MatlabSim2PythonSim.Simulation( simDir )

# instantiate the A-RAIM class
ARAIM = Integrity.MultiHypothesisSolutionSeperation()

# instantiate the WLS class
WLS = Integrity.MultiHypothesisSolutionSeperation()

# generate list of all GPS & GALILEO sats
satList = list(range(1,33))
satList.extend(list(range(71,106)))

# set URE sig to 0.5 meters and sig URA to 0.75 meters
sigURE =0.5*np.ones( len( satList ) )
sigURA = 0.75*np.ones( len( satList ) )
maxBiasNom = 0.5*np.ones( len( satList ) )
# default the p_fail to example in Blanch et al 
pSats = 1.0e-4*np.ones( len( satList ) )
pConstellation =[1.0e-5, 1.0e-5]
constellationList = ['GPS','GAL']

# set up the ISM
ARAIM.ISMUpdate(sigURA, sigURE, maxBiasNom, pSats, pConstellation, \
		satList, constellationList )

WLS.ISMUpdate(sigURA, sigURE, maxBiasNom, pSats, pConstellation, \
		satList, constellationList )

# provide an initial user position & clock bias
x,L = np.shape(data.GNSS.nSat)
ARAIM.initPos(data.GNSS.ECEFNom, 3000000)
xwls,Lwls = np.shape(data.GNSS.nSat)
WLS.initPosWLS(data.GNSS.ECEFNom, 3000000)
#26000
while i < 2000:
	i = i + 10
	print ("i:")
	nSat = data.GNSS.nSat[0,i]

	#failSat1 = []
	#failSat2 = []

	chanceFault = random.uniform(0,1)
	print(chanceFault)

	#data.GNSS.carrierSmoothPR[100,6] = data.GNSS.carrierSmoothPR[100,6] + 10

	if (chanceFault < .1):
		satFault1 = random.randint(0,19)
		failSat1 = satFault1
		#failSat1.append( satFault1 )
		data.GNSS.carrierSmoothPR[i,satFault1] = data.GNSS.carrierSmoothPR[i,satFault1] +30
		
		print("Error injected at i = %d on satellite %d"%(i,satFault1))

		chanceFault2 = random.uniform(0,1)
		print(chanceFault2)
		if (chanceFault2 < .1):
			satFault2 = random.randint(0,19)
			failSat2 = satFault2
			data.GNSS.carrierSmoothPR[i,satFault2] = data.GNSS.carrierSmoothPR[i,satFault2] + 30
			print ("Error injected at i = %d on satellite %d"%(i,satFault2))
		else:
			failSat2 = None
	else:
		failSat1 = None
		failSat2 = None
		
	print(failSat1)
	print(failSat2)	

	HPL, VPL = ARAIM.ARAIM( nSat, \
		data.GNSS.satXYZ[0:nSat,:,i], \
		data.GNSS.carrierSmoothPR[i,0:nSat], \
		data.GNSS.svID[i,0:nSat])

	OMCWLS, G0WLS = WLS.WLSCheck( nSat, \
		data.GNSS.satXYZ[0:nSat,:,i], \
		data.GNSS.carrierSmoothPR[i,0:nSat], \
		data.GNSS.svID[i,0:nSat], failSat1, failSat2)
	
	nSats= ARAIM.numSat(data.GNSS.svID[i,0:nSat].reshape(-1).tolist() \
		,constellationList)
	usrPos = ARAIM.getPos()	
	usrPosWLS = WLS.getPosWLS()	
	print ("Last Position Solution %5.4f %5.4f %5.4f"%(usrPos[0],usrPos[1],usrPos[2] ))
	print ("Number of GPS %d / Number of GAL %d"%(nSats[0],nSats[1]))
	print ("Fault Modes Evaluated %d"%(ARAIM.getNFaultModes()))
	print ("Time %d,  HorizontalPL %5.4f, Vertical PL %5.4f"%(i, HPL, VPL))
	thresh4mAt95prcnt, thresh10mFaultFree, EMT= ARAIM.getAccuracyLimits()
	print ("0.95 Vertical Accuracy: %5.4f, 10e-7 Fault Free Accuracy: %5.4f, EMT: %5.2f "%( \
		thresh4mAt95prcnt, \
		thresh10mFaultFree, EMT ))
	print (" All in-view Chi-Square Stat %5.5f Eval %5.5f East Fail %d North Fail %d Vertical Fail %d"%(ARAIM.chiSqStat, \
		ARAIM.chiSqEval, ARAIM.numOfModesFailed[0], ARAIM.numOfModesFailed[1], ARAIM.numOfModesFailed[2]))
	#print "Chi-Square list "%(ARAIM.chilist)
	stringE = "East " + str(ARAIM.testStats[0]).strip('[]')
	stringN = "North " + str(ARAIM.testStats[1]).strip('[]')
	stringV = "Vertical " + str(ARAIM.testStats[2]).strip('[]')
	#print(stringE)
	#print(stringN)
	#print(stringV)
	#f= open("TestfailuresTEST.txt", 'a')
	#f.write(str(i)  + str(ARAIM.numOfModesFailed) + '\n')
	#print(ARAIM.numOfModesFailed)
	#f.close()
	#f= open("OtherTestDataTestRunT100.txt", 'a')
	#f.write('New epoch ' + str(i) + '\n')
	#f.write('User Position: ' + str(usrPos) + '\n')
	#f.write('Number of Sats GPS/GAL: ' + str(nSats) + '\n')
	#f.write('Fault Modes:' + str(ARAIM.getNFaultModes()) + '\n')
	#f.write('0.95 Vertical Accuracy/ 10e-7 Fault Free Accuracy/ EMT: ' + str(thresh4mAt95prcnt) + str(thresh10mFaultFree) + str(EMT))
	#f.write('HPL VPL: ' + str(HPL) + ' | ' + str(VPL))
	#f.close()
	#f= open("TEST2ENV.txt", 'a')
	#f.write('New epoch ' + str(i) + '\n')
	#f.write('\n' + str(stringE) + '\n')
	#f.write(str(stringN) + '\n')
	#f.write(str(stringV) + '\n')
	#f.close()

	#f= open("UsrPosTestRunT100.txt", 'a')
	#f.write(str(usrPos) + '\n')
	#f.close()			                
	

