#!/usr/bin/env python

"""
   This is a python module that implements GNSS integrity-monitoring algorithms.
   __author__='Jason N. Gross'
   __email__='Jason.Gross@mail.wvu.edu'
"""

import numpy as np
import scipy.linalg
import scipy.interpolate
import scipy.stats
import itertools
import math

# wvu png modules, will need AFIT equivalents
import ARAIM.Transform as navutils
import ARAIM.Const as Const
import ARAIM.Model as Model

# Using this function so ARAIM.utils is not needed
def find(a, func):
	""" 
	implementation of a matlab-style find function for python
	source: http://stackoverflow.com/questions/5957470/matlab-style-find-function-in-python

	"""
	return [i for i, val in enumerate(a) if func(val)]

class KFIntegrityMonitor():
	
	""" 
	This class is the implementation of the Kalman Filter Integrity
	Monitoring algorithm that is derived in the following paper:
	Joerger, Mathieu, and Boris Pervan. 
	"Kalman filter-based integrity monitoring against sensor faults." 
	Journal of Guidance, Control, and Dynamics 36.2 (2013): 349-361.
	
	all units are in SI- MKS: Meter, Kilogram, Seconds	
	"""

	def __init__( self, printer =True, 
			   probabilityOfFailure= 1.0e-4, 
			   continuityRiskRequirement = 8.0e-6, 
			   integrityRisk= 1.0e-7, 
			   alertLimit=10.0):

		""" 
		    initialize KF Integrity Monitoring Algorithm
		    The default values for probabilities are taken from:
		    RTCA Special Committee 159, 
		    "aMinimum Aviation System Performance Standards 
		    for the Local Area Augmentation System
		    (LAAS)," ( RTCA/DO-245), RTCA, Inc., Washington, D.C., 2004,
		    Appendix D.
		"""
                
		self.probabilityOfFailure = probabilityOfFailure
		self.continuityRiskRequirement = continuityRiskRequirement
		self.integrityRisk = integrityRisk
		self.alertLimit = alertLimit

		self.dataRate = 1.0 # defaults is 1 Hz
		self.nEpochHist = 100 # number of epochs over which to evaluate RSOS
		
		# initialize the printer flag
		# if the printer flaf is True 
		# a lot of content will be displayed to screen 
		# for debugging  purposes
		self.printer = printer
		
		# some convienience vars
		self.hashString ='#--------------------------------------------------------#'
		
		# initialize Residual Sum of Squares (RSOS)
		self.histRSOS = []
		self.totalRSOS = 0.0

		# these terms define the generalized chi-square distribution of the test statistic
		# see eq. 36 of Joerger and Pervan KF-IM paper
		self.alphas = []
		self.ys = []

		if(self.printer):
			print(self.hashString)
			print("Initializing GNSS Integrity Class")
			print("Apriori Probability of Failure:	 %e"%( self.probabilityOfFailure ))
			print("Continuity Risk Requirement:	 %e"%( self.continuityRiskRequirement ))
			print("Integrity Risk Requirement:	%e"%( self.integrityRisk ))
			print("Alert Limit:	%f"%( self.alertLimit ))
			

	def postfitResiduals(self, z, x, H):
		""" inputs:
			z - KF Observed Minus Computed (OMC) vector, i.e., the pre-fit residuals
				<np.matrix> m by 1 
			x - KF state vector
				<np.matrix> n by 1
			H - KF observation matrix
				<np.matrix> m by n
		     output:
		      currentPostfitRes - KF postfit residuals
				<np.matrix> m by 1
		"""

		self.currentPostfitRes = []
		self.currentPostfitRes = z - H*x
		L=len(self.currentPostfitRes)
		if (self.printer):
			print(self.hashString)
			for i in range(L):
				print (' Postfit Residuals %d of %d %15.6f'%(i+1,L,self.currentPostfitRes[i]))
			
	def accumulateRSOS(self,R):
		""" 
		Evaluate current RSOS add to the array that is keeping a running history, 
		removing elements if necessary 
		inputs:
			R - KF Measurement Error Covariance
		"""
		Rinv= np.linalg.inv( R )
		self.currentRSOS = float( self.currentPostfitRes.T*Rinv*self.currentPostfitRes )
		if ( self.printer ):
			print( self.hashString )
			print ("Current Epoch RSOS: %f"%( self.currentRSOS ))

		if ( len( self.histRSOS )< self.nEpochHist ):
			self.histRSOS.append( self.currentRSOS )
		else:
			self.histRSOS.pop( 0 )
			self.histRSOS.append( self.currentRSOS )

		self.totalRSOS = np.sum( self.histRSOS )

		if( self.printer ):
			print( self.hashString )
			print ("Total RSOS %f"%( self.totalRSOS ))


	def setPrinterFlag( printerFlag ):
		self.printer = printerFlag
	
	def setIntegrityRiskRequirement( integrityRisk ):
		self.integrityRisk = integrityRisk

	def setAlertLimit( alertLimit ):
		self.alertLimit = alertLimit

	def residualCovarianceMatrix( H,K,R,Pminus ):
		"""
		Calculate the covariance matrix of the Kalman Filter Postfit Residuals 
		inputs:
			H - GNSS Observation matrix
			K - Kalman Gain Matrix
			R - KF Assumed Measurement Error Covariance
		"""
		
		n=H.shape[1] # H is m by n, get n ( length of state vector )
		I = np.eye( n )

		# eq 28 of Joerger's Paper
		self.residualCov = (I - H*K)*R*(I - H*K).T - H*(I - K*H)*Pminus*(I - K*H).T*H.T

		if( self.printer ):
			print( self.hashString )
			print ("Residual Covariance Matrix")
			print( self.residualCov )



	def determineNoncentralChiSqrDistributionAlphaParams( self, R ):
		"""
			Determine the alpha parameters of the noncentralized 
			chi-squared distribution that represents the current time
			test statistic (i.e. the 'currentRSOS')
			This is eqs. 29-30 of Joerger and Pervan's paper.
		inputs:
			R - KF Assumed Measurement Error Covariance Matrix 
		"""

		measErrCovSqrtInv = scipy.linalg.sqrtm( np.linalg.inv( R ) )
		sqrtResidualCov = scipy.linalg.sqrtm( self.residualCovariance )

		[U, s, V] = np.linalg.svd( measErrCovSqrtInv*sqrtResidualCov )

		# append the whole array of alphas
		self.alphas.append(s)

	


class MultiHypothesisSolutionSeperation():
	"""
		This class is an implementation of the Multiple Hypothesis Solution Seperation
		RAIM algorithm presented in:
		Blanch, Juan, et al. "An optimized multiple hypothesis RAIM algorithm for vertical 
		guidance." Proceedings of the 20th International Technical Meeting of the 
		Satellite Division of The Institute of Navigation (ION GNSS 2007). 2001.
		and More recently:
		Blanch, Juan, et al. "Baseline advanced RAIM user algorithm and 
		possible improvements." 
		Aerospace and Electronic Systems, IEEE Transactions on 51.1 (2015): 713-732
		 all units are in SI- MKS: Meter, Kilogram, Seconds
	"""

	def __init__(self, printer=True):

		# initialize the printer flag
		# if the printer flag is True
		# a lot of content will be displayed to screen for debugging  purposes
		self._printer = printer
		# some convienience vars
		self._hashString ='#--------------------------------------------------------#'
		self._constants = Const.Const()
		# define the algorithm constants (Table II in Blanch et al, 2015)
		self._pHMI = 1.0e-7	# total integrity budget
		self._pHMIVert = 9.0e-8  # integrity budget for the vertical component
		self._pHMIHoriz = 1.0e-8 # integrity budget for the horizontal compoenent
		self._pThresh = 9.0e-8   # threshold for the integrity risk coming from the unmonitored faults
		self._pFA = 4.0e-6	# continuity budget allocated to disruptions b/c false alert
		self._pFAVert = 3.9e-6	# continuity budget allocated to the vertical mode
		self._pFAHoriz = 9.0e-8	# continuity budget allocated to the horizontal mode
		self._pFAChiSqr = 1.0e-8 # continuity budget allocated to the chi squared test
		self._tolPL = 5.0e-2 	# tolerance for the computation of the protection level {meters}
		self._kACC = 1.96	# number of std dev used for the accuracy formula
		self._kFF = 5.33		# number of std dev used for the 1.0e-7 FF vert pos err.
		self._pEMT = 1.0e-5	# probabaility used for the caluclation of EMT
		self._tRecov = 600	# min. time a prev. excl sat reamins of of the all-in-view pos sol (quaratine) {sec.}
		self._maxNFaultConsidered = 10 # number of simulatneous faults considered in the tests
		# constants used for Error Models, and detectors
		self.__defineGalieoUserErrorModel()
		self.__defineGPSUserErrorModelFactor()
		self.__defineBoundsForDeterminationOfNFaults()
		self._dx0 = np.matrix( np.zeros(5) )
		self._dx0wls = np.matrix( np.zeros(5) )
		self._dx0N = np.matrix( np.zeros(5) )
		# Set of parameters ot make this somewhat comparable with ARAIM
		# If simple is on, it just sets all covariance to the same
		# value: self._Cint_simples
		self._simpleCovMode = False
		self._Cint_simple = 10.**2.0
		# If simple is on, it just sets all fault probabilities to the same value
		self._simpleFaultMode = False
		self._fault_prob = 0.01
		# To enable making it similar to FGO, allow the VPL to just turn off the effect of bias
		self._useBiasForPL = True
		
		
	
	def __defineGalieoUserErrorModel(self):
		"""
		Emperical nominal elevation angle dependent model for the Galileo constellation 
		user error. Defined in Table A.I of Blanch et al, 2015.
		"""
		if self._printer:
			print(self._hashString)
			print ("Defining Galileo Nominal URE from Emperical Function ")

		elevationAngles = np.arange(0,95,5)
		galileoUserError = np.array([0.4529, 0.4529, 0.3553, 0.3063, 0.2638, 0.2593, 0.2555, 0.2504, 
			0.2438, 0.2396, 0.2359, 0.2339, 0.2302, 0.2295, 
			0.2278, 0.2297, 0.2310, 0.2274, 0.2277] )
		self._galileoErrorModelInterpolant = scipy.interpolate.interp1d(elevationAngles, galileoUserError, kind='cubic')
	

	def __defineGPSUserErrorModelFactor(self):
		"""
		Determine the frequency dependent scaling factor within Eq. 57(1) of Blanch et. al. 2015
		"""
		if self._printer:
			print(self._hashString)
			print ("Definining GPS Nominal URE ")
		self._gpsErrorModelScaleFactor = np.sqrt( ( self._constants.f1**4.0 + self._constants.f2**4.0 ) \
				/ ( self._constants.f1**2.0 + self._constants.f2**2.0 )**2.0 )

	def __defineBoundsForDeterminationOfNFaults(self):
		"""
		Determine the probability bounds for assessing the max # of faults that must be tested explicilty for 
		failure
		Eqs. 75-78 of Blanch et al. 2015
		"""

		self._lowerBoundNFaultMax = []
		self._upperBoundNFaultMax = []
		self._lowerBoundNFaultMax.append(0)
		self._upperBoundNFaultMax.append(self._pThresh)

		for nFail in range(1,self._maxNFaultConsidered):
			self._lowerBoundNFaultMax.append( ( math.factorial(nFail) * self._pThresh )**(1.0/nFail ) )
			self._upperBoundNFaultMax.append( ( math.factorial(nFail+1) * self._pThresh )**(1.0/(nFail+1)) )

	def __determineNFaultMax(self):
		""" 
		Determine N-Fault Max, by evaluating Eq. 78 of Blanch et al. 2015 
		"""
		for i in range(self._maxNFaultConsidered):
			if ( ( self._pFaultTotal > self._lowerBoundNFaultMax[i] ) and    \
				( self._pFaultTotal <= self._upperBoundNFaultMax[i] )):
				self._NFaultMax = i
		if self._printer:
			print(self._hashString)
			print ("N Fault Max Determined to be %d"%(self._NFaultMax))
		return

		
	def __determinePossibleFaultModeCombinations(self, svID):
		"""
		Determine all the possible subsets of sats exlcuing possible failures
		start with {n;k} = n!/(k!*(n-k)!) where 
			n is (number of failure events)
			k is (number of failure events - NFaultMax)
			n-k = NFaultMax
		"""
		self._faultModes=[]
		self._faultModeCompliments=[]
		self._pFaultModes=[]
		events = svID.reshape(-1).tolist()
		events.extend( self._constellationsTracked )
		# for now just set pUnobservable = 0.0
		self._pUnobservable = 0.0
		combinations = []
		# start with Nfault max, and step down to single faults
		for nFaults in range(self._NFaultMax, 0, -1 ):
			for combo in itertools.combinations(events, self._nSatsTracked + 
					self._nConstTracked - nFaults ):
				combinations.append(combo)
		
		for combination in combinations:
			# make sure at least one const present in mode
			nConstMissing = 0
			constDropped = []
			for i in range( len(self._constellationsTracked) ):
				if not find( combination, lambda x: x == self._constellationsTracked[i] ):
					nConstMissing = nConstMissing + 1.0
					constDropped.append( self._constellationsTracked[i] )
			# consider the fact that if a const fails, we have to drop all of those sats
			nSatsOrig = len([x for x in combination if not isinstance(x,str)]) # number of sats in this mode
			nSatsDropped =0 # number of sats dropped from mode due to constellation failure / to be computed
			nConstOrig = len([x for x in combination if isinstance(x,str)]) # number of constellation in this mode
			for const in constDropped:
				for fault in combination:
					# if fault is a sat, and it belongs to this const
					if not isinstance( fault, str ) :
						if const == self.__svid2constellation( fault ):
							nSatsDropped = nSatsDropped +1

			nParmsToSolve = 3 + nConstOrig # 3 position + 1 bias for each const
			nObsAvailable = nSatsOrig - nSatsDropped


			if (nConstMissing >= self._nConstTracked ):
				# print ("Deleting Failure Mode from Consideration becayse all constellations would be failed ")
				p, combo = self.__probabilityOfFaultMode( combination, svID)
				self._pUnobservable = self._pUnobservable + p
				continue

			elif (nParmsToSolve > nObsAvailable ):
				print ( "Deleting Failure Mode due to insufficient obs left Nparms %d, NobsAvail %d"%(nParmsToSolve, nObsAvailable))
				p, combo = self.__probabilityOfFaultMode( combination, svID)
				self._pUnobservable = self._pUnobservable + p
				continue
			else:
				#print combination
				self._faultModes.append( combination )
				p, complimentEvents = self.__probabilityOfFaultMode( combination, svID )
				self._pFaultModes.append( p )
				self._faultModeCompliments.append( complimentEvents )
			

		 
		self._NFaultModes = len( self._faultModes )
		# may need to prune more failure modes
		
	def __probabilityOfFaultMode(self,faultMode, svID):
		""" 
		Sum up the probability of a specific fault mode, given the list of elements in the mode
		and the ISM contents
		"""

		pFaultMode = 1.0
		# determine the probability of mutually independent events occuring
		allEvents = []
		allEvents.extend( self._constellationsTracked )
		allEvents.extend( svID )
		allEvents = set( allEvents )
		faultMode = set( faultMode )
		eventsNotInMode = allEvents - faultMode
		if self._simpleFaultMode:
			pFaultMode = self._fault_prob
		else:
			for event in list(eventsNotInMode):
				# determine if the fault is a sat fault or constellation fault
				# if it is a number, this refers to a PRN
				if not isinstance( event, str ) :
					ind = find( self._satList, lambda x: x == event )[0]
					pFaultMode = pFaultMode *self._pSats[ind] 
				else: # otherwise, it is a constellation failure
					ind = find( self._constellationList, lambda x: x == event )[0]
					pFaultMode = pFaultMode * self._pConstellation[ind]

		return pFaultMode, eventsNotInMode

	def __probabilityOfMultipleFaultNotMonitored(self):
		"""
		Determine the upper bound on the probability of multiple faults
		that are not monitored, this is the upper bount evaluated for r= NFaultMax +1 
		This is Eq. 74 from Blanch et al 2015
		"""
		r = self._NFaultMax + 1

		self._pMultiFaultNotMonitored = ( ( self._pFaultTotal )**r ) / math.factorial( r )

	def __probabilityFaultNotMonitored(self):
		"""
		Determine the integrity risk of the modes not explicitly monitored.
		This is Equation 11 of Blanch et al 2015
		"""
		self._pFaultNotMonitored = self._pMultiFaultNotMonitored + self._pUnobservable

	
	def __calculateFaultModeWeightMatrix(self, faultMode, svID):
		"""
		For given fault mode, zero out elements of the weight matrix that are failed.
		Eq. 12 of Blanch et al. 2015
		inputs:
			faultMode - list of satellites and constellations included in this solution mode
			svID - list of the all in view satellites
		output:
			a Weight matrix that zeros out certain elements of missing sats/constellations
		"""
		if self._simpleCovMode:
			WfaultMode = np.ones(len(svID)) * 1./self._Cint_simple
		else:
			# intiialize the fault mode weight elements to inverse of the all in view covariance 
			WfaultMode = 1./np.array(self._Cint)
		deletedIndices = []
		# loop over all sats in list, zeroing elements not found in the fault mode
		for i in range( len(svID) ) :
			if not svID[i] in faultMode:
			# if not find( faultMode, lambda x: x==svID[i]):
				deletedIndices.append(i)
				WfaultMode[i] = 0.0

		# now loop over contellationsTracked, zeroing weights if sat is in a constellation not present
		for i in range( len(self._constellationsTracked) ):
			# if we can't find a constellation, zero the contribution of all sats from that constellation
			if not self._constellationsTracked[i] in faultMode:
			# if not find( faultMode, lambda x: x== self._constellationsTracked[i] ):
				for j in range(len(svID)):
					# we know that self.constellationsTracked_[i] is an excluded one,
					# so zero the weight of all the sats that match this constellation
					if self._constellationsTracked[i] == self.__svid2constellation( svID[j] ):
						deletedIndices.append(i)
						WfaultMode[j] = 0.0
		
		# make sure same index wasnt collected twice in above computations
		deletedIndices = list( set( deletedIndices ) )
		# return the weight matrix
		return np.matrix( np.diag( WfaultMode ) ) , deletedIndices


	def __reshapeGeometryMatrixForMissingConstellation(self, faultMode, G):
		"""
		If an entire costellation is missing from a particular solution fault mode,
		then we need to remove the column of the geomtry matrix that includes
		the clock bias partials for that constellation.
		
		inputs:
			faultMode - lists of satellites and constellations included in this solution 
			G - the all in view geometry matrix
		
		outputs:
			G - the geometry matrix with reshaped to no include columns with clock partials 
			    for excluded sats
		"""
		colsToDelete  = []
		
		# first test the easy case. look for the constellations that explicily not listed in the
		# fault mode vector
		for i in range( len(self._constellationsTracked) ):
			# if we cant find this constellation, remove associated column from G
			if not find( faultMode, lambda x: x == self._constellationsTracked[i] ):
				# get the column number, by finding index of this constellation
				# in the list of constellations tracked + 3 for the position partials
				colsToDelete.append( 3.0 + find( self._constellationsTracked, \
					lambda x: x == self._constellationsTracked[i] )[0] )
				
		# now loop through all sats in fault mode, and make sure the cover all contellations
		constInFaultMode = []
		for event in faultMode:
			# only check sats in fault mode, so look for events thats are numbers
			if( isinstance( event, (float, int, complex) ) ):
				constInFaultMode.append( self.__svid2constellation( event ) )

		# get the list of unique sats tracked, these are names of constellations
		constInFaultMode = list( set( constInFaultMode ) )

		# check that each constellation in lists tracked is covered by the sats in the fault mode
		for constellation in self._constellationsTracked:
			if not find( constInFaultMode, lambda x: x == constellation ):
				colsToDelete.append( 3.0 + find( self._constellationsTracked,lambda x:x == constellation)[0] )

		# get unique cols to delete
		colsToDelete = list( set( colsToDelete ) )
		
		G = np.array( G )
		if len( colsToDelete ) == 1:
			
			return np.matrix( np.delete(G, int(colsToDelete[0]) ,1) )
		else:
			G = np.array( G )
			# handle case of deleteing more than one column, where the matrix size is shrinking
			# such that old column sizes loose their meaning
			colsToDelete.sort()
			n=0
			for col in colsToDelete:

				G = np.delete(G, int(col-n) , 1)
				n = n + 1
		
			return np.matrix(G)


	def __gpsErrorModelSigMP(self, elv):
		"""
		Multipath GPS user error estimate dependent on elevation angle
		Eq 57(2) of Blanch et. al. 2015.
		input:
			elv - satellite elevation angle from the user's perspective {rad.}
		output:
			estimate of error standard deviation {meters}
		"""
		elv = elv*(180.0/np.pi)
		return 0.13 + 0.53*np.exp( -elv / 10.0 )

	def __gpsErrorModelSigNoise(self, elv):
		"""
		Sigma Noise GPS user error based on elevation angle.
		Eq. 57(3) of Blanch et. al. 2015.
		input:
			elv - satellite elevation angle from the user's perspective {rad.}
		output:
			estimate of error standard deviation {meters}
		"""
		elv = elv*(180.0/np.pi)
		return 0.15 + 0.43*np.exp( -elv / 6.9 )

	def __gpsUserErrorModel(self, elv):
		"""
		GPS Error model, Equation 57(1) of Blanch et. al. 2015.
		input:
			elv - satellite elevation angle from the user's perspective {rad.}
		output:
			estimate of GPS Signal In Space User Error {meters}
		"""
		return self._gpsErrorModelScaleFactor*np.sqrt( self.__gpsErrorModelSigNoise(elv)**2.0 + self.__gpsErrorModelSigMP(elv)**2.0 )

	def __tropErrorModel(self, elv):
		"""
		Trop Error Model model used in baseline ARAIM Blanch et. al. 2015. Eq. 58
		
		input:
			elv - satellite elevation angle from the user's perspective {rad.}
		output:
			estimate of the tropospheric delay {meters}
		"""

		return 0.12* ( 1.001 / np.sqrt( 0.002001 + ( np.sin( ( elv ) ) )**2.0 ) )

	def __calculateAllInViewCovariance(self, satsXYZ, usrPos, svID):
		"""
		Compute the pseudorange covariance matrix for integrity Cint and Cacc
		inputs:
			satsXYZ - ECEF XYZ list of satellite positions {np.array}(#Sats by 3)
			usrPos  - ECEF XYZ user positions {np.array}(1 by 3)
			svID - satellite vehicle identifier of 
				observations with order consistent with satsXYZ
				convention adopted from Septentrio "SBF Reference Guide V 1.9.0"
				1 - 37 is GPS and corresponds to PRN
				38 - 61 is GLONASS and is slot + 37
				71 - 106 is GALILEO anf is PRN + 70
				62 used for unknown GLONASS
		outputs:
			Cint - diagonal elements of the nominal error model used for integrity
			Cacc - diagonal elements of the nominal error model used for continuity
		"""

		self._Cint = []
		self._Cacc = []

		self._Nsat = np.shape( satsXYZ )[0]

		for i in range( self._Nsat ):
			if self._simpleCovMode:
				self._Cint.append( self._Cint_simple )
				self._Cacc.append( self._Cint_simple )
				continue
			elv = navutils.calcElAz( satsXYZ[i] , usrPos )[0]
			if svID[i] <= 37:
				userError = self.__gpsUserErrorModel( elv )
			elif svID[i] >= 71 and svID[i] <= 106:
				userError = self.__galileoUserErrorModel( elv )
			else:
				print ("Only GPS and Galileo Are Currently Supported ")

			tropErr = self.__tropErrorModel( elv )
			self._Cint.append( self._sigURA[i]**2.0 + tropErr**2.0 + userError**2.0 )
			self._Cacc.append( self._sigURE[i]**2.0 + tropErr**2.0 + userError**2.0 )


	def getNFaultModes( self ):
		"""
			Return number of fault modes evaluated
		"""
		return self._NFaultModes

	def getAccuracyLimits( self ):
		""" 
			Return: 
				-4m 95% bound
				-10m 1e-7 bound
		"""
		return self._thresh4mAt95prcnt, self._thresh10mFaultFree, self._EMT

	def getPos( self ):
		"""
			Return user position solution
		"""
		return self._usrPos
	
	def getUpdatedPos( self ):
		"""
			Return user position solution with dx0 applied
		"""
		enuDelta = np.array(-self._dx0[0,0:3]).ravel()
		return navutils.enu2xyz(enuDelta,self._usrPos)

	def getPosWLS( self ):
		"""
			Return user position solution formed by WLS
		"""
		return self._usrPosWLS

	def getPosN( self ):
		"""
			Return user position solution formed by WLS
		"""
		return self._usrPosN

	def initPos( self, usrPos = [0,0,0] , clockNom = 0.0):
		"""
			Initialize the ECEF position of the receiver and the clock bias
		"""
		
		self._usrPos = usrPos 
		self._clockNom = clockNom

	def initPosWLS( self, usrPosWLS = [0,0,0] , clockNomWLS = 0.0):
		"""
			Initialize the ECEF position of the receiver and the clock bias
		"""
		
		self._usrPosWLS = usrPosWLS 
		self._clockNomWLS = clockNomWLS

	def initPosN( self, usrPosN = [0,0,0] , clockNomN = 0.0):
		"""
			Initialize the ECEF position of the receiver and the clock bias
		"""
		
		self._usrPosN = usrPosN 
		self._clockNomN = clockNomN


	def __observationMatrix( self, G, W ):
		""" 
			Calculate the observation matrix, H, given the geometry matrix, G
			
			Input:
				G - geometry matrix
				W - based on self._Cint for weighting the solution, 
				    but calculated elsewhere
			Output:
				observation matrix
		"""

		return np.linalg.inv(G.T*W*G)*G.T*W

	def __parityMatVec( self, H, OMC ):
		""" 
			Calculate the parity vector and matrix
			From Eq. 19 & 20 of
			Joerger, M., et al. "Integrity Risk and Continuity Risk for Fault Detection 
			and Exclusion Using Solution Separation ARAIM." Proceedings of the 
			26th International Technical Meeting of The Satellite Division of the 
			Institute of Navigation (ION GNSS 2013). 2013.
		"""
		u, s, v = np.linalg.svd(H.T)
		rank = (s > 1E-5*s[0]).sum() #magic number alert!!! :)
		QT = v[rank:].T.copy()

		p = QT.T*OMC

		self.residualChiSquareStat = p.T*p
		return  p, QT.T

	def WLSN(self, nSat, satsXYZ, prObs, svID):

		"Method for doing WLS"

		# Step #0 apply nominmal to delta
		count = 0
		print("In the WLS Method")
		print(self._dx0N[0,0:3])
		print(self._usrPosN)
		enuDeltaN = []
		enuDeltaN.append( -self._dx0N[0,0] )
		enuDeltaN.append( -self._dx0N[0,1] )
		enuDeltaN.append( -self._dx0N[0,2] )
		self._usrPosN = navutils.enu2xyz(enuDeltaN,self._usrPosN)
		print(self._usrPosN)
		print ("WLSN User Position %5.4f %5.4f %5.4f"%(self._usrPosN[0],self._usrPosN[1],self._usrPosN[2]))
		self._clockNomN = self._clockNomN - self._dx0N[0,3]
		

		# Step #2 All in view solution

		self.__probabilityOfNoFault(svID)
		self.__probabilityOfFault(svID)
		G0N = self.__determineGeometryMatrix(satsXYZ, self._usrPosN,  svID)
		
		OMCN = self.__determineObservedMinusComputed( prObs, satsXYZ, \
				self._usrPosN, self._clockNomN )
		

		
		self.__calculateAllInViewCovariance(satsXYZ, self._usrPosN, svID)
			

		enuCov0, W0 = self.__wlsAllInViewMatrix3( G0N, svID, OMCN )

		print("Completed the WLS method")

		return OMCN, G0N


	def WLSCheck(self, nSat, satsXYZ, prObs, svID, failSat1, failSat2):

		"Method for doing WLS with direct removal of satellite data without ARAIM"

		# Step #0 apply nominmal to delta
		count = 0
		print("In the WLSCheck Method")
		print(self._dx0wls[0,0:3])
		print(self._usrPosWLS)
		enuDeltawls = []
		enuDeltawls.append( -self._dx0wls[0,0] )
		enuDeltawls.append( -self._dx0wls[0,1] )
		enuDeltawls.append( -self._dx0wls[0,2] )
		self._usrPosWLS = navutils.enu2xyz(enuDeltawls,self._usrPosWLS)
		print(self._usrPosWLS)
		print ("WLS User Position %5.4f %5.4f %5.4f"%(self._usrPosWLS[0],self._usrPosWLS[1],self._usrPosWLS[2]))
		self._clockNomWLS = self._clockNomWLS - self._dx0wls[0,3]
		

		# Step #2 All in view solution

		self.__probabilityOfNoFault(svID)
		self.__probabilityOfFault(svID)
		G0WLS = self.__determineGeometryMatrix(satsXYZ, self._usrPosWLS,  svID)
		
		OMCWLS = self.__determineObservedMinusComputed( prObs, satsXYZ, \
				self._usrPosWLS, self._clockNomWLS )
		

		if (failSat1 != None):
			Gx, OMCx, svIDx, count, satsXYZx = self.SatRemove(failSat1, G0WLS, OMCWLS, svID, count, satsXYZ)

			self.__calculateAllInViewCovariance(satsXYZx, self._usrPosWLS, svIDx)
			

			enuCov0, W0 = self.__wlsAllInViewMatrix2( Gx, svIDx, OMCx )

		if (failSat2 != None):

			Gx2, OMCx2, svIDx2, count, satsXYZx2 = self.SatRemove(failSat2, Gx, OMCx, svIDx, count, satsXYZx) # jason added fix for 2nd failure 1/11/25

			self.__calculateAllInViewCovariance(satsXYZx2, self._usrPosWLS, svIDx2)
			

			enuCov0, W0 = self.__wlsAllInViewMatrix2( Gx2, svIDx2, OMCx2 )

		else:

			self.__calculateAllInViewCovariance(satsXYZ, self._usrPosWLS, svID)
			

			enuCov0, W0 = self.__wlsAllInViewMatrix2( G0WLS, svID, OMCWLS )

		print("Completed the WLSCheck method")

		return OMCWLS, G0WLS

		

	def ARAIM(self, nSat,  satsXYZ, prObs, svID):
		"""
		Main method for running ARAIM
		inputs:
			nSat - Number of Sats for this epoch of data
			satsXYZ - ECEF satellite locations for this epoch
			prObs - psuedorange observations
			svID - sat id numbers
		outputs:
			HPL - horizontal protection limit (m)
			VPL - vertical protection limit (m)
			exclude_list - list of satelites that are excluded
		"""
		
		count = 0
		count2 = 0
		count3 = 0
		#Time = 10
		minChi = 0
		ExcIndex = 0
		exclude_list = []
		# Step #0 apply nominmal to delta
		# print(self._dx0[0,0:3])
		# print(self._usrPos)
		enuDelta = []
		enuDelta.append( -self._dx0[0,0] )
		enuDelta.append( -self._dx0[0,1] )
		enuDelta.append( -self._dx0[0,2] )
		self._usrPos = navutils.enu2xyz(enuDelta,self._usrPos)
		# print(self._usrPos)
		self._clockNom = self._clockNom - self._dx0[0,3] #was ,1.  I think that was an error

		# Step #1 determine Cint and Cacc
		self.__calculateAllInViewCovariance(satsXYZ, self._usrPos, svID)
		self.__probabilityOfNoFault(svID)
		self.__probabilityOfFault(svID)


		# Step #2 All in view solution
		G0 = self.__determineGeometryMatrix(satsXYZ, self._usrPos,  svID)
		
		OMC = self.__determineObservedMinusComputed( prObs, satsXYZ, \
				self._usrPos, self._clockNom )
		
		enuCov0, b0, W0 = self.__wlsAllInViewMatrix( G0, svID, OMC, count, ExcIndex )

		# Step #3 Determine Fauilts to be Monitored
		self.__determineNFaultMax()
		self.__determinePossibleFaultModeCombinations(svID)
		self.__probabilityOfMultipleFaultNotMonitored()
		self.__probabilityFaultNotMonitored()

		# Step #4 Fault Tolerant Positions and Std Dev.
		dxk = np.zeros([self._NFaultModes, 4 ])
		enuCovk = np.zeros( [ self._NFaultModes , 3 ] ) 
		enuCovSSk = np.zeros( [ self._NFaultModes , 3 ] )
		bk = np.zeros( [ self._NFaultModes , 3 ] )
		Tk = np.zeros( [ self._NFaultModes , 3 ] )
		pFaultk = np.zeros( self._NFaultModes )
		
		self.numOfModesFailed = [ 0, 0, 0] # re-initialize mode failure counters
		self.testStats = [ [], [], [] ]
		
		
		for k in range( self._NFaultModes ):
			
			

			faultMode = self._faultModes[k]	
			pFaultk[k], faultCompliment= self.__probabilityOfFaultMode(faultMode, svID )
			dxk[k,:], enuCovk[k,:], enuCovSSk[k,:], bk[k,:] \
				=self.__evaluateWLSFaultModeSolnSep( G0, W0, OMC, \
				svID, faultMode, count, ExcIndex )
		
			status, Tk[k,:], count2 = self.__evalFaultModeThresholdTest( faultMode,\
					faultCompliment, \
					dxk[k,:], enuCovSSk[k,:], count2) 

			

			minChi, ExcIndex = self.minSatPick(G0, W0, OMC, svID, faultMode, minChi, ExcIndex )

		#f= open("TestfailuresTestRunT1003.txt", 'a')
		#f.write(str(self.numOfModesFailed) + '\n')
		#print(ARAIM.numOfModesFailed)
		#f.close()

		
		if (count2 != 0):

			exclude_list.append(ExcIndex)

			# print(self.numOfModesFailed)

			# print ("Faulted Satellite: %d"%(ExcIndex))

			# print ("E Fail: %d N Fail: %d V Fail: %d"%(self.numOfModesFailed[0],self.numOfModesFailed[1],self.numOfModesFailed[2]))

			# print ("Fault Sat Plus Failures: %d %d %d %d"%(ExcIndex,self.numOfModesFailed[0],self.numOfModesFailed[1],self.numOfModesFailed[2]))
			

			Gx, Wx, OMCx, svIDx, count, satsXYZx = self.SatExc(ExcIndex, G0, W0, OMC, svID, count, satsXYZ)


			self.__calculateAllInViewCovariance(satsXYZx, self._usrPos, svIDx)
			self.__probabilityOfNoFault(svIDx)
			self.__probabilityOfFault(svIDx)


			enuCov0, b0, W0 = self.__wlsAllInViewMatrix( Gx, svIDx, OMCx, count, ExcIndex )

			self.__determineNFaultMax()
			self.__determinePossibleFaultModeCombinations(svIDx)
			self.__probabilityOfMultipleFaultNotMonitored()
			self.__probabilityFaultNotMonitored()

			dxk = np.zeros([self._NFaultModes, 4 ])
			enuCovk = np.zeros( [ self._NFaultModes , 3 ] ) 
			enuCovSSk = np.zeros( [ self._NFaultModes , 3 ] )
			bk = np.zeros( [ self._NFaultModes , 3 ] )
			Tk = np.zeros( [ self._NFaultModes , 3 ] )
			pFaultk = np.zeros( self._NFaultModes )


			minChi = 0
			ExcIndex = 0

			#self.numOfModesFailed = [ 0, 0, 0] # re-initialize mode failure counters
			#self.testStats = [ [], [], [] ]

			for k in range( self._NFaultModes ):

				faultMode = self._faultModes[k]
				
					
				pFaultk[k], faultCompliment= self.__probabilityOfFaultMode(faultMode, svIDx )
				dxk[k,:], enuCovk[k,:], enuCovSSk[k,:], bk[k,:] \
					=self.__evaluateWLSFaultModeSolnSep( Gx, Wx, OMCx, \
					svIDx, faultMode, count, ExcIndex )
	
				status, Tk[k,:], count3 = self.__evalFaultModeThresholdTest( faultMode,\
					faultCompliment, \
					dxk[k,:], enuCovSSk[k,:], count3)
			
				minChi, ExcIndex = self.minSatPick(Gx, Wx, OMCx, svIDx, faultMode, minChi, ExcIndex )

			#f= open("TestfailuresTestRunT1003.txt", 'a')
			#f.write(str('Faulted ') + str(self.numOfModesFailed) + '\n')
			#print(ARAIM.numOfModesFailed)
			#f.close()

			#f= open("dx0sTestRunT100.txt", 'a')
			#f.write(str(self._dx0[0,0:3]) + '\n')
			#print(ARAIM.numOfModesFailed)
			#f.close()

			chiStatus = self.__evalChiSquareOfAllInViewSoln(Gx, OMCx, count, ExcIndex)
 
		
			EPLup,EPLlow  = self.__calculateComponentProtectionLimit( b0, enuCov0, \
				bk, enuCovk, Tk, pFaultk, 0)
			NPLup, NPLlow = self.__calculateComponentProtectionLimit( b0, enuCov0, \
				bk, enuCovk, Tk, pFaultk, 1)
			VPL, VPLlow = self.__calculateComponentProtectionLimit( b0, enuCov0, \
				bk, enuCovk, Tk, pFaultk, 2)
			HPL = np.sqrt( EPLup**2 + NPLup**2 )

			self.__calcuateFFSolnAccuracyAndEMT(Gx, Wx, Tk, pFaultk, count, ExcIndex)
			count2 = 0
			#Time = Time + 10

			if (count3 != 0):
				exclude_list.append(ExcIndex)
				# print(self.numOfModesFailed)

				# print ("Faulted Satellite: %d"%(ExcIndex))

				# print ("E Fail: %d N Fail: %d V Fail: %d"%(self.numOfModesFailed[0],self.numOfModesFailed[1],self.numOfModesFailed[2]))

				# print ("Fault Sat Plus Failures: %d %d %d %d"%(ExcIndex,self.numOfModesFailed[0],self.numOfModesFailed[1],self.numOfModesFailed[2]))
			

				Gx2, Wx2, OMCx2, svIDx2, count, satsXYZx2 = self.SatExc(ExcIndex, Gx, Wx, OMCx, svIDx, count, satsXYZx)



				self.__calculateAllInViewCovariance(satsXYZx2, self._usrPos, svIDx2)
				self.__probabilityOfNoFault(svIDx2)
				self.__probabilityOfFault(svIDx2)

				enuCov0, b0, W0 = self.__wlsAllInViewMatrix( Gx2, svIDx2, OMCx2, count, ExcIndex )

				self.__determineNFaultMax()
				self.__determinePossibleFaultModeCombinations(svIDx2)
				self.__probabilityOfMultipleFaultNotMonitored()
				self.__probabilityFaultNotMonitored()

				dxk = np.zeros([self._NFaultModes, 4 ])
				enuCovk = np.zeros( [ self._NFaultModes , 3 ] ) 
				enuCovSSk = np.zeros( [ self._NFaultModes , 3 ] )
				bk = np.zeros( [ self._NFaultModes , 3 ] )
				Tk = np.zeros( [ self._NFaultModes , 3 ] )
				pFaultk = np.zeros( self._NFaultModes )


				minChi = 0
				ExcIndex = 0
				#self.numOfModesFailed = [ 0, 0, 0] # re-initialize mode failure counters
				#self.testStats = [ [], [], [] ]

				for k in range( self._NFaultModes ):

					faultMode = self._faultModes[k]
				
					
					pFaultk[k], faultCompliment= self.__probabilityOfFaultMode(faultMode, svIDx2 )
					dxk[k,:], enuCovk[k,:], enuCovSSk[k,:], bk[k,:] \
						=self.__evaluateWLSFaultModeSolnSep( Gx2, Wx2, OMCx2, \
						svIDx2, faultMode, count, ExcIndex )
	
					status, Tk[k,:], count3 = self.__evalFaultModeThresholdTest( faultMode,\
						faultCompliment, \
						dxk[k,:], enuCovSSk[k,:], count3)
			
					minChi, ExcIndex = self.minSatPick(Gx2, Wx2, OMCx2, svIDx2, faultMode, minChi, ExcIndex )

				#f= open("TestfailuresTestRunT1003.txt", 'a')
				#f.write(str('Faulted ') + str(self.numOfModesFailed) + '\n')
			#print(ARAIM.numOfModesFailed)
				#f.close()

				#f= open("dx0sTestRunT100.txt", 'a')
				#f.write(str(self._dx0[0,0:3]) + '\n')
			#print(ARAIM.numOfModesFailed)
				#f.close()

				chiStatus = self.__evalChiSquareOfAllInViewSoln(Gx2, OMCx2, count, ExcIndex)
 
		
				EPLup,EPLlow  = self.__calculateComponentProtectionLimit( b0, enuCov0, \
					bk, enuCovk, Tk, pFaultk, 0)
				NPLup, NPLlow = self.__calculateComponentProtectionLimit( b0, enuCov0, \
					bk, enuCovk, Tk, pFaultk, 1)
				VPL, VPLlow = self.__calculateComponentProtectionLimit( b0, enuCov0, \
					bk, enuCovk, Tk, pFaultk, 2)
				HPL = np.sqrt( EPLup**2 + NPLup**2 )

				self.__calcuateFFSolnAccuracyAndEMT(Gx2, Wx2, Tk, pFaultk, count, ExcIndex)
				count3 = 0

		else:

			#f= open("dx0sTestRunT100.txt", 'a')
			#f.write(str(self._dx0[0,0:3]) + '\n')
			# print(self.numOfModesFailed)
			# print ("E Fail: %d N Fail: %d V Fail: %d"%(self.numOfModesFailed[0],self.numOfModesFailed[1],self.numOfModesFailed[2]))
			#f.close()			

			chiStatus = self.__evalChiSquareOfAllInViewSoln(G0, OMC, count, ExcIndex)
 
		
			EPLup,EPLlow  = self.__calculateComponentProtectionLimit( b0, enuCov0, \
				bk, enuCovk, Tk, pFaultk, 0)
			NPLup, NPLlow = self.__calculateComponentProtectionLimit( b0, enuCov0, \
				bk, enuCovk, Tk, pFaultk, 1)
			VPL, VPLlow = self.__calculateComponentProtectionLimit( b0, enuCov0, \
				bk, enuCovk, Tk, pFaultk, 2)
			HPL = np.sqrt( EPLup**2 + NPLup**2 )
			
			self.__calcuateFFSolnAccuracyAndEMT(G0, W0, Tk, pFaultk, count, ExcIndex)

			#Time = Time + 10

		

		return HPL, VPL, exclude_list



	def ISMUpdate(self, sigURA, sigURE, maxBiasNom, \
			pSats, pConstellation, satList, \
			constellationList=['GPS','GLO','GAL']):

		"""
		Update the algorithm parameters that come from the Integrity Support Message (ISM)
		these are assuming a baseline ISM from Table I in (Blanch et. al.; 2015)
		inputs:
			sigURA - std. dev. of the clock and ephemeris
				error for Integrity {np.array}(#Sat by 1)
			sigURE - std. dev. of the clock and 
				ephemeris error for Accuracy/Continuity {np.array}(#Sat by 1)
			maxBiasNom - max nominal bias for sat. i 
				used for integrity {np.array}(#Sat by 1)
			pSats - prior probability of fault in 
				sat i per approach {np.array}(#Sat by 1)
			pConstellation - prior probability of 
				fault affecting more than 1 sat in 
				constellation {np.array}(#Cont by 1)
			satList - list of svID numbers that correspond 
				to order of errors & probabilies with mapping for 
				GPS (1-37), GLONASS (38-61), GALILEO (71-106)
			constellationList - list of constellations 
				to give order of pConstellation with 
				entries 'GPS', 'GLO' , 'GAL', 'BDS'
		"""
		self._sigURA = sigURA
		self._sigURE = sigURE
		self._maxBiasNom = maxBiasNom
		self._pSats = pSats
		self._pConstellation = pConstellation
		self._satList = satList
		self._constellationList = constellationList


	def __probabilityOfNoFault(self,svID):
		"""
		Determine the probability of no fault given the ISM contents 
		Implementation of Eq. 68 in Blanch 2015
		"""
		self._nSatsTracked = len(svID)

		if self._simpleFaultMode:
			self._nConstTracked = 1
			self._constellationsTracked = ['GPS']
			self._pNoFault = (1- self._fault_prob)**len(svID)
			return

		[self._nConstTracked, self._constellationsTracked] =	\
				self.__determineNumberOfConstThisEpoch(svID)

		# initialize to 1
		self._pNoFault = 1.0

		for i in range(self._nSatsTracked):
			# find index in ISM vector that corresponds to each tracked prn
			ind = find( self._satList, lambda x: x == svID[i] )[0]
			
			# multiplicatively reduce probability of no faults
			self._pNoFault = self._pNoFault * ( 1.0 - self._pSats[ind] )

		for i in range(self._nConstTracked):
			# find index of ISM vector that corresponds to
			# each tracked constellation
			ind = find( self._constellationList, \
					lambda x: x == self._constellationsTracked[i] )[0]
			
			self._pNoFault = self._pNoFault * ( 1.0 - self._pConstellation[ind] )


	def __probabilityOfFault(self, svID):
		"""
		Determine the total probability of a fault via summation.
		This is finding the 'u' for the phiThreshFunction in Eqs. 75-77
		in Blanch et al. 2015.
		"""
		if self._simpleFaultMode:
			self._pFaultTotal = self._fault_prob
			return
		
		# initialize to zero
		self._pFaultTotal = 0.0

		# save the total number of inpendent events 
		# individual sat failures + constellation wide failures
		self._NIndependentEvents = self._nSatsTracked + self._nConstTracked 

		for i in range(self._nSatsTracked):
			# find the index of this sat in the ISM vector that corresponse to this prn
			ind = find( self._satList, lambda x: x == svID[i])[0]
			# sum up
			self._pFaultTotal = self._pFaultTotal + self._pSats[ind]

		for i in range(self._nConstTracked):
			# find the index of the ISM vector that corresponds to the probability of constellation failure
			ind = find( self._constellationList, lambda x: x == self._constellationsTracked[i] )[0]
			self._pFaultTotal = self._pFaultTotal + self._pConstellation[ind] 

	
	def __svid2constellation( self, svID):
		""" 
		Given an svID number, return the constellation, 'GPS','GLO','GAL' or 'BDS'
		"""
		if find( [svID], lambda x: x <= 37):
			return 'GPS'
		if find( [svID], lambda x: x> 37 and x<=61):
			return 'GLO'
		if find( [svID], lambda x: x>70 and x<=106):
			return 'GAL'

	def __determineNumberOfConstThisEpoch( self, svID ):
		"""
		Given a list of sv-ID's tracked, determine how many constellations are present.
		"""

		nConst = 0
		constellationsTracked = []
		
		if find( svID, lambda x: x <= 38):
			# gps is being tracked
			nConst = nConst + 1
			constellationsTracked.append('GPS')

		if find( svID, lambda x: x>37 and x<=61):
			# glonass is being tracked
			nConst = nConst + 1
			constellationsTracked.append('GLO')

		if find( svID, lambda x: x>70 and x<=106):
			# galileo is being tracked
			nConst = nConst + 1
			constellationsTracked.append('GAL')

		## !!! NEED TO ADD BEIDOU Support

		

		return nConst, constellationsTracked

	def __galileoUserErrorModel(self, elv ):
		"""
		Calculate the elevation angle from the user's perspective 
		in order to call the emperical error model
		
		inputs:
			elv - satellite elevation angle from the user's perspective {rad.}
		
		"""
		
		return self._galileoErrorModelInterpolant( elv*180.0/np.pi )


	
	def __determineGeometryMatrix(self, satsXYZ, usrPos,  svID):
		"""
		Given a list of satellite poisitions and their PRN, determine the geometry matrix
		in ENU with individual clock bias partials for each constellation.
		
		inputs:
			satsXYZ - ECEF XYZ list of satellite positions {np.array}(#Sats by 3)
			usrPos  - ECEF XYZ user positions {np.array}(1 by 3)
			svID - satellite identifier of observations with order consistent with satsXYZ
			nConst - number of constellation's being processed this epoch
		"""
	
		# initialize the geometry matrix
		nSat = len( svID )
		G = np.zeros( ( nSat, self._nConstTracked + 3 ) )
		for j in range(nSat):
			satENU = navutils.xyz2enu(satsXYZ[j,:], usrPos)
			G[j,0:3] = satENU/np.linalg.norm(satENU)

			# find the correct colum for the clock bias partial
			# by comparingthe constellation of this sat
			# against the list of constellations tracked this epoch
			ind = find( self._constellationsTracked, \
					lambda x: x == self.__svid2constellation( svID[j] ) )[0] + 3
			G[j,ind] = 1.0


		return G

		

	def __determineObservedMinusComputed(self, measuredPR, satsXYZ, usrPos, clockNom):
		"""
		Determine the OMC prefit residuals 
		
		inputs:
			measuredPR - receiver's psuedorange measurements
			satsXYZ - ECEF XYZ list of satellite positions {np.array}(#Sats by 3)
		        usrPos  - ECEF XYZ user positions {np.array}(1 by 3)
		output:
			OMC - a vector of receiver measured minus nominal predicted GNSS 
			      psuedoranges {np.array}(# Sat by 1)
		"""	

		nSat = len(measuredPR)
		computed = np.zeros( nSat )
		omc = np.zeros( nSat )
		
		for i in range( nSat ):
			computed[i] = np.linalg.norm( np.array( satsXYZ[i,:] ) - np.array( usrPos ) ) \
					+ clockNom 
					# + Model.dryTrop(satsXYZ[i,:],usrPos)
			omc[i] = measuredPR[i] - computed[i]
		
		return np.matrix(omc).T

	def __wlsAllInViewMatrix(self, G , svID, OMC, count, ExcIndex):
		"""
		Determine weighted least squares
		for all satellite in view S Matrix, save as class member.
		
		inputs:
			G - geometry matrix in ENU with 
			    different clock bias partials for each constellation
		outputs:
			stores (G^T*W*G)^(-1) as class parameter
			returns 
				envCov of all in view solution
				b worst case impact from bias of all in view
				W weight matrix of all in view
		"""
		# the diagonal elements of the weighting matrix are
		# the inverse of the error covariance
		Wdiag = 1./np.array( self._Cint )
		W = np.matrix( np.diag( Wdiag ) )

		#if (count > 0):
				#W = np.delete(W, ExcIndex, 0)
				#W = np.delete(W, ExcIndex, 1)

		
		self._AllInViewInvGTWG = np.linalg.inv( G.T*W*G )
		
		enuCov = np.diag( self._AllInViewInvGTWG )
		S0=self._AllInViewInvGTWG*G.T*W
		x,y=np.shape(S0)
		self._dx0[0,0:x] = (S0 *OMC).T
		# print(self._dx0[0,0:x])
		# determine worst-case impact from biases Eq. 16 from Blanch
		b = np.zeros(3)
		for q in range(3):
			for i in range( len(svID) ):
				ind = find( self._satList, lambda x: x == svID[i] )[0]
				b[q] = b[q] + abs(S0[q,i])*self._maxBiasNom[ind]
		
		return enuCov, b, W

	def __wlsAllInViewMatrix2(self, G , svID, OMC):
		"""
		Determine weighted least squares
		for all satellite in view S Matrix, save as class member.
		
		inputs:
			G - geometry matrix in ENU with 
			    different clock bias partials for each constellation
		outputs:
			stores (G^T*W*G)^(-1) as class parameter
			returns 
				envCov of all in view solution
				b worst case impact from bias of all in view
				W weight matrix of all in view
		"""
		# the diagonal elements of the weighting matrix are
		# the inverse of the error covariance
		Wdiag = 1./np.array( self._Cint )
		W = np.matrix( np.diag( Wdiag ) )


		
		self._AllInViewInvGTWG = np.linalg.inv( G.T*W*G )
		
		enuCov = np.diag( self._AllInViewInvGTWG )
		S0=self._AllInViewInvGTWG*G.T*W
		x,y=np.shape(S0)
		self._dx0wls[0,0:x] = np.array(S0*OMC).T
		print(self._dx0wls[0,0:x])
		
		return enuCov, W

	def __wlsAllInViewMatrix3(self, G , svID, OMC):
		"""
		Determine weighted least squares
		for all satellite in view S Matrix, save as class member.
		
		inputs:
			G - geometry matrix in ENU with 
			    different clock bias partials for each constellation
		outputs:
			stores (G^T*W*G)^(-1) as class parameter
			returns 
				envCov of all in view solution
				b worst case impact from bias of all in view
				W weight matrix of all in view
		"""
		# the diagonal elements of the weighting matrix are
		# the inverse of the error covariance
		Wdiag = 1./np.array( self._Cint )
		W = np.matrix( np.diag( Wdiag ) )


		
		self._AllInViewInvGTWG = np.linalg.inv( G.T*W*G )
		
		enuCov = np.diag( self._AllInViewInvGTWG )
		S0=self._AllInViewInvGTWG*G.T*W
		x,y=np.shape(S0)
		self._dx0N[0,0:x] = np.array(S0*OMC).T
		print(self._dx0N[0,0:x])
		
		return enuCov, W

	def __evaluateWLSFaultModeSolnSep(self, G, W, OMC, svID, faultMode, count, ExcIndex ):
		"""
		Use rank one updates to determine an WLS S matrix 
		for a fault mode
		inputs: 
		 	G - geometry matrix in ENU w/ different clock bias partials
			for each GNSS constellation
		"""
		 # check if the fault mode is only one sat (This is likely majority of cases)
		Wk, delIndices = self.__calculateFaultModeWeightMatrix(faultMode, svID)
		S0=self._AllInViewInvGTWG*G.T*W
		if len( delIndices ) == 1:
			delInd = delIndices[0]
			
			G=np.matrix(G)
			# perform a rank 1 update to determine the positon soln difference
			dx =  (self._AllInViewInvGTWG*G[delInd,:].T*W[delInd,delInd])/(1.0 -  \
					G[delInd,:]*W[delInd,delInd]*self._AllInViewInvGTWG*G[delInd,:].T)* \
					( OMC[delInd,0] - G[delInd,:]*self._AllInViewInvGTWG*G.T*W*OMC )
		
			invGTWGk = self._AllInViewInvGTWG + ( self._AllInViewInvGTWG*(G[delInd,:].T* \
					W[delInd,delInd]*G[delInd,:])*self._AllInViewInvGTWG ) / \
					( 1.0 - G[delInd,:]*W[delInd,delInd]*self._AllInViewInvGTWG*G[delInd,:].T )
			
			enuCov = np.diag( invGTWGk )
		

			
			#if (count > 0):
				#Wk = np.delete(Wk, ExcIndex, 0)
				#Wk = np.delete(Wk, ExcIndex, 1)
			
			Sk = invGTWGk*G.T*Wk
			

		else:
		# if more than one, just revert to standard, less efficient, calculation
			Gk =  self.__reshapeGeometryMatrixForMissingConstellation(faultMode, G)
			#print(Gk.shape)
			#if (count > 0):
				#Wk = np.delete(Wk, ExcIndex, 0)
				#Wk = np.delete(Wk, ExcIndex, 1)
			invGTWGk = np.linalg.inv( Gk.T*Wk*Gk )
			Sk =invGTWGk*Gk.T*Wk
			
			enuCov = np.diag( invGTWGk )

			x,y=np.shape(Sk)
			S0=self._AllInViewInvGTWG*G.T*W
			dx = ( Sk - S0[0:x,0:y] ) * OMC



		# determine the ENU covariance of the solution seperation, dx
		enuCovSS = np.zeros(3)
		Cacc = np.matrix( np.diag( self._Cacc ) )

		#if (count > 0):
				#Cacc = np.delete(Cacc, ExcIndex, 0)
				#Cacc = np.delete(Cacc, ExcIndex, 1)
		x,y=np.shape(Sk)
		for q in range(3):
			e = np.zeros(x)
			e[q] = 1.0
			e = np.matrix(e)
			enuCovSS[q] = e*( Sk - S0[0:x,0:y] )*Cacc*( Sk - S0[0:x,0:y] ).T*e.T


		
		# determine worst-case impact from biases Eq. 16 from Blanch
		b = np.zeros(3)
		for i in range( len(svID) ):
			ind = np.where(self._satList == svID[i])[0][0]
			# ind = find( self._satList, lambda x: x == svID[i] )[0]
			for q in range(3):
				b[q] = b[q] + abs(Sk[q,i])*self._maxBiasNom[ind]

		enuCov=enuCov[0:3]
		deltaX=np.zeros(4)
		deltaX[0]=dx[0,0]
		deltaX[1]=dx[1,0]
		deltaX[2]=dx[2,0]
		deltaX[3]=dx[3,0]
		return deltaX, enuCov, enuCovSS, b

	def minSatPick(self, G, W, OMC, svID, faultMode, minChi, ExcIndex ):

		
		Wk, delIndices = self.__calculateFaultModeWeightMatrix(faultMode, svID)

		#print(delIndices)

		chiall = OMC.T*( W - W*G*np.linalg.inv( G.T*W*G )*G.T*W )*OMC

		if len( delIndices ) == 1:

			delInd = delIndices[0]

			#print(delInd)

			#G=np.matrix(G)

			#Gk =  self.__reshapeGeometryMatrixForMissingConstellation(faultMode, G)


			GTrans = np.array([G[delInd,:]])
			

			chiupdate = chiall - (W[delInd,delInd]/( 1.0 - G[delInd,:]*W[delInd,delInd]*self._AllInViewInvGTWG*GTrans.T))*(( OMC[delInd,0] - G[delInd,:]*self._AllInViewInvGTWG*G.T*W*OMC )**2)

			#print(chiupdate)

			chisubtract = (W[delInd,delInd]/( 1.0 - G[delInd,:]*W[delInd,delInd]*self._AllInViewInvGTWG*GTrans.T ))*(( OMC[delInd,0] - G[delInd,:]*self._AllInViewInvGTWG*G.T*W*OMC )**2)

			if chiupdate < minChi or minChi == 0:

				minChi = chiupdate
				ExcIndex = delInd
				# print(chiupdate)
				#print(chiall)
				#print(chisubtract)
				# print(ExcIndex)
				# print ("Chi-Squared Stat for Sat %d is %5.4f"%(ExcIndex,chiupdate))
			else:
				minChi = minChi
				ExcIndex = ExcIndex
				# print(chiupdate)
				# print(delInd)
				# print ("Chi-Squared Stat for Sat %d is %5.4f"%(delInd,chiupdate))

			

				
			#print(ExcIndex)

		#else:

			#chiupdate = OMC.T*(Wk-(Wk*Gk*Sk))*OMC

			#if chiupdate < minChi or minChi == 0:

			#	minChi = chiupdate
			#else:
			#	minChi = minChi
		


		return minChi, ExcIndex

			
	def SatExc(self, ExcIndex, G, W, OMC, svID, count, satsXYZ):

		# print(ExcIndex)
		#print(svID(ExcIndex))
		
		# print(G.shape)
		# print(W.shape)
		# print(OMC.shape)
		# print(len(svID))
		# print(satsXYZ.shape)

		Gx = np.delete(G, ExcIndex, 0)
		OMCx = np.delete(OMC, ExcIndex, 0)
		Wx = np.delete(W, ExcIndex, 0)
		Wx = np.delete(Wx, ExcIndex, 1)
		svIDx = np.delete(svID, ExcIndex, 0)
		satsXYZx = np.delete(satsXYZ, ExcIndex, 0)

		# print(Gx.shape)
		# print(Wx.shape)
		# print(OMCx.shape)
		# print(len(svIDx))
		# print(satsXYZx.shape)
		
		count = count + 1

		return Gx, Wx, OMCx, svIDx, count, satsXYZx

	def SatRemove(self, failSat, G, OMC, svID, count, satsXYZ):

		print(failSat)
		#print(svID(ExcIndex))
		
		print(G.shape)
		print(OMC.shape)
		print(len(svID))
		print(satsXYZ.shape)

		Gx = np.delete(G, failSat, 0)
		OMCx = np.delete(OMC, failSat, 0)
		svIDx = np.delete(svID, failSat, 0)
		satsXYZx = np.delete(satsXYZ, failSat, 0)

		print(Gx.shape)
		print(OMCx.shape)
		print(len(svIDx))
		print(satsXYZx.shape)
		
		count = count + 1

		return Gx, OMCx, svIDx, count, satsXYZx

	
	def __evalFaultModeThresholdTest(self, faultMode, faultCompliment, dx, enuCovSS, count2):
		"""
		Implementation of Solution Seperation Threshold Tests.
		Eqs. 18-21 from Blanch et al. 2015
		inputs:
			faultMode - description of faultMode
			faultCompliment - events that compliment the fault mode
			dx - solution seperation between all in view and fail mode
			enuCovSS - enu components of the covariance of solution seperation for this mode
		output:
			status
				False is No Failure Detected
				True is Failure is Detected
		"""

		Kfa = []
		Kfa.append( ( scipy.stats.norm.isf( self._pFAHoriz / (4.0*self._NFaultModes) ) ) )
		Kfa.append( ( scipy.stats.norm.isf( self._pFAHoriz / (4.0*self._NFaultModes) ) ) )
		Kfa.append( ( scipy.stats.norm.isf( self._pFAVert / (2.0*self._NFaultModes) ) ) )
	
		#print(Kfa)
		#print(enuCovSS)

		T=[None,None,None]
		for q in range(3):
			T[q] = Kfa[q]*np.sqrt( enuCovSS[q] )
			tau = abs( dx[q] ) / ( T[q] ) 
			self.testStats[q].append(tau)
			if tau > 1.0:
				#print "Fail %f > 1.0, \n Events included in Failure -->"%(tau)
				#print( faultCompliment )
				self.numOfModesFailed[q] += 1 
				#return True , T
				count2 = count2+1

			
		return False, T, count2

	def __evalChiSquareOfAllInViewSoln(self, G, OMC, count, ExcIndex ):
		"""
		Calculate the Chi-Squared statistics for the all-in-view solution.
		Eq 22 of Blanch et al. 2015
		If all solution seperation tests pass, but chi-square test fails,
		there is still a failure likely, so a potection level cannot be evaluated
		inputs:
			G - all in view geometry matrix with position partials in ENU
			OMC - difference between observed and computed psuedoranges
		output:
			status
				True if Chi Square Test Fails ( failure detected )
				False if Chi Square Test passes ( no failure )
		"""


		Wacc = np.matrix( np.diag( self._Cacc ) )

		#if (count > 0):
				#Wacc = np.delete(Wacc, ExcIndex, 0)
				#Wacc = np.delete(Wacc, ExcIndex, 1)

		self.chiSqStat = OMC.T*( Wacc - Wacc*G*np.linalg.inv( G.T*Wacc*G )*G.T*Wacc )*OMC
		dof = self._nSatsTracked - 3 - self._nConstTracked
		self.chiSqEval = scipy.stats.chi2.pdf( self.chiSqStat, dof )
		
		return ( self.chiSqEval > ( 1.0 - self._pFAChiSqr ) )


	def __calculateComponentProtectionLimit(self, b0, enuCov0, bk, enuCovk, Tk, pFaultk, qIndex):
		"""
		Calculate for the Vertical/Horizontal Protection Limit (VPL) using the linear approximation
		(NOT ITERATIVE USING BISECTION)
		Appendix B.A) in Blanch et al 2015.
		inputs:
			b0 - worst-case positioning impact from biases for all in view solution {1,3}
			enuCov0 - estimated covariance of all in view position solution {1,3}
			bk - worst-case positioning impact from biases for fault modes {k,3}
			enuCovk - estimated covariance of fault mode solution {k,3}
			Tk - solution seperation test statistics {k,3}
			pFaultK - probability of each fault mode {1,k}
			qIndex - component requested 
				0 - East
				1 - North
				2 - Vertical
		"""
		# Eq. 62
		if qIndex == 2:
			pHMI = self._pHMIVert
		else:
			pHMI = self._pHMIHoriz

		pHMIAdj = pHMI*( 1.0 - self._pFaultNotMonitored / ( self._pHMIVert + self._pHMIHoriz ) )
		
		temp1= ( scipy.stats.norm.isf( pHMIAdj / 2.0 ))*np.sqrt(enuCov0[qIndex]) + b0[qIndex] 

		temp2 = []

	
		for k in range( len( Tk ) ):
			temp2.append( ( scipy.stats.norm.isf( pHMIAdj / pFaultk[k] ) )*np.sqrt(enuCovk[k,qIndex]) + Tk[k,qIndex] + bk[k,qIndex] )
		
		
		PLlowInit = max( temp1, max(temp2) )

		
		temp1=  scipy.stats.norm.isf( pHMIAdj / (2.0*(self._NFaultMax + 1.0)))*np.sqrt(enuCov0[qIndex]) + b0[qIndex]
		
		temp2 = []

		
		for k in range( len( Tk ) ):
			
			temp2.append( ( scipy.stats.norm.isf( pHMIAdj / ( pFaultk[k]*(self._NFaultMax +1.0 ) ) ) )*np.sqrt(enuCovk[k,qIndex]) + Tk[k,qIndex] + bk[k,qIndex] )

		PLupInit = max( temp1, max(temp2) )
		PLupBound = PLlowInit + ( pHMI - self.__pExceed( PLlowInit, b0, enuCov0,bk, enuCovk, Tk, pFaultk, qIndex ) )*( PLupInit - PLlowInit )/ \
				( self.__pExceed(PLupInit, b0, enuCov0,bk, enuCovk, Tk, pFaultk, qIndex) \
				- self.__pExceed(PLlowInit, b0, enuCov0,bk, enuCovk, Tk, pFaultk, qIndex) )

		PLlowBound = PLlowInit + (np.log( pHMIAdj ) - np.log( self.__pExceed( PLlowInit, b0, enuCov0,bk, enuCovk, Tk, pFaultk, qIndex ) ) )* \
				(PLupInit - PLlowInit)/( self.__pExceed( PLupInit, b0, enuCov0,bk, enuCovk, Tk, pFaultk, qIndex ) - \
				self.__pExceed( PLlowInit, b0, enuCov0,bk, enuCovk, Tk, pFaultk, qIndex ) )


		return PLupBound, PLlowBound


	def __pExceed( self, VPL, b0, enuCov0, bk, enuCovk, Tk, pFaultk, qInd):
		"""
		determine pExceed ( used within protection limit calcs )
		Eq. 60 of Blanch et al 2015
		"""
		pExceed1 =  2.0*( scipy.stats.norm.sf( (VPL - b0[qInd])/np.sqrt(enuCov0[qInd]) ) )
		
		pExceed2 = 0.0
		for k in range( len( Tk ) ):
			pExceed2 = pExceed2 + pFaultk[k]*( scipy.stats.norm.sf( ( VPL - Tk[k,qInd] -bk[k,qInd] )/ np.sqrt(enuCovk[k,qInd]) ) )
	
		return (pExceed1 + pExceed2)

	def numSat(self,svID, constellations):
		"""
		Determine number of sats per constellation in specifiec list of constellations
		inputs:
			svID - sat list
			constellations - list of constellations 'GPS','GAL','GLO', etc.
		"""
		numSat = []
		for const in constellations:
			n = 0
			for i in range(len(svID)):
				if const == self.__svid2constellation(svID[i]):
					n = n+1
			numSat.append(n)
		return numSat

	def __calcuateFFSolnAccuracyAndEMT(self, G, W, Tk, pFaultk, count, ExcIndex):
		""" 
		1) Evaluate the standard deviation of the vertical position solution using Eq. 27 
		2) Determine the accuracy bound to test 95% at 4m
		3) Determine the accuracy bound to test 99.99999% 10^(-7)m
		4) Determine the effective mointor threshold (EMT) to test 15m at 99.999%
		"""

		
		S0=self._AllInViewInvGTWG*G.T*W

		e = np.matrix( np.zeros( np.shape( S0 )[0] ) )
		e[0,2] = 1.0
		Cacc = np.matrix( np.diag( self._Cacc ) )

		#if (count > 0):
		#		Cacc = np.delete(Cacc, ExcIndex, 0)
		#		Cacc = np.delete(Cacc, ExcIndex, 1)

		sigVACC = np.sqrt( e*S0*Cacc*S0.T*e.T )

		# print("sigVACC: ")
		# print(sigVACC)

		self._thresh4mAt95prcnt = self._kACC*sigVACC # less stringent requirement
		self._thresh10mFaultFree = self._kFF*sigVACC # more stringent requirement

		Temt= []
		for k in range( self._NFaultModes ):
			if ( pFaultk[k] >= self._pEMT ):
				Temt.append( Tk[k,2] )
		self._EMT = max( Temt )
