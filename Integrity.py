#!/usr/bin/env python

"""
   This is a python module that implements many GNSS integrity-monitoring algorithms.

   __author__='Jason N. Gross'
   __email__='Jason.Gross@mail.wvu.edu'
"""

import numpy as np
import scipy.linalg
import scipy.interpolate

# wvu png modules, will need AFIT equivalents
import Transform as navutils
import Const

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
			print "Initializing GNSS Integrity Class"
			print "Apriori Probability of Failure:	 %e"%( self.probabilityOfFailure )
			print "Continuity Risk Requirement:	 %e"%( self.continuityRiskRequirement )
			print "Integrity Risk Requirement:	%e"%( self.integrityRisk )
			print "Alert Limit:	%f"%( self.alertLimit )
			

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
				print ' Postfit Residuals %d of %d %15.6f'%(i+1,L,self.currentPostfitRes[i])
			
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
			print "Current Epoch RSOS: %f"%( self.currentRSOS )

		if ( len( self.histRSOS )< self.nEpochHist ):
			self.histRSOS.append( self.currentRSOS )
		else:
			self.histRSOS.pop( 0 )
			self.histRSOS.append( self.currentRSOS )

		self.totalRSOS = np.sum( self.histRSOS )

		if( self.printer ):
			print( self.hashString )
			print "Total RSOS %f"%( self.totalRSOS )


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
			print " Residual Covariance Matrix "
			print( self.residualCovariance )



	def determineNoncentralChiSqrDistributionAlphaParams( R ):
		"""
			Determine the alpha parameters of the noncentralized 
			chi-squared distribution that represents the current time
			test statistic (i.e. the 'currentRSOS')

			This is eqs. 29-30 of Joerger and Pervan's paper.

		inputs:
			R - KF Assumed Measurement Error Covariance Matrix 
		"""

		measErrCovSqrtInv = scipy.linalg.sqrtm( np.linalg.inv( R ) )
		sqrtResidualCov = scipy.linalg.sqrtm( self.resdiualCovariance )

		[U, s, V] = np.linalg.svd( measErrCovSqrtInv*sqrtResidualCov )

		# append the whole array of alphas
		self.alphas.append(s)

	#def determineWorstCaseFaultVector( )


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
		Aerospace and Electronic Systems, IEEE Transactions on 51.1 (2015): 713-732.
	"""

	def __init__(self, printer=True):

		# define the algorithm constants (Table II in Blanch et al, 2015)
		self.pHMI = 1.0e-7	# total integrity budget
		self.pHMIVert = 9.0e-8  # integrity budget for the vertical component
		self.pHMIHoriz = 1.0e-8 # integrity budget for the horizontal compoenent
		self.pThresh = 9.0e-8   # threshold for the integrity risk coming from the unmonitored faults
		self.pFA = 4.0e-6	# continuity budget allocated to disruptions b/c false alert
		self.pFAVert = 3.9e-6	# continuity budget allocated to the vertical mode
		self.PFAHoriz = 9.0e-8	# continuity budget allocated to the horizontal mode
		self.pFAChiSqr = 1.0e-8 # continuity budget allocated to the chi squared test
		self.tolPL = 5.0e-2 	# tolerance for the computation of the protection level {meters}
		self.kACC = 1.96	# number of std dev used for the accuracy formula
		self.kFF = 5.33		# number of std dev used for the 1.0e-7 FF vert pos err.
		self.pEMT = 1.0e-5	# probabaility used for the caluclation of EMT
		self.tRecov = 600	# min. time a prev. excl sat reamins of of the all-in-view pos sol (quaratine) {sec.}
		
		# constants used for Galileo Error Models
		self.defineGalieoUserErrorModel_(self)
		
		# initialize the printer flag
		# if the printer flaf is True
		# a lot of content will be displayed to screen for debugging  purposes
		self.printer = printer

		# some convienience vars
		self.hashString ='#--------------------------------------------------------#'

	def defineGalieoUserErrorModel_(self):
		"""
		Emperical nominal elevation angle dependent model for the Galileo constellation 
		user error. Defined in Table A.I of Blanch et al, 2015.
		"""

		elevationAngles = np.arange(5,95,5)
		galileoUserError = np.array([0.4529, 0.3553, 0.3063, 0.2638, 0.2593, 0.2555, 0.2504, 0.2438, 0.2396, 0.2359, 0.2339, 0.2302, 0.2295, 0.2278, 0.2297, 0.2310, 0.2274, 0.2277] )
		galileoUserErrorModelInerpolant_ = scipy.interpolate.interp1d(elevationAngles, galileoUserError, kind='cubic')
	
	
	def defineGPSUserErrorModelFactor_(self):

		"""
		Determine the frequency dependent scaling factor within Eq. 57(1) of Blanch et. al. 2015
		"""
		self.gpsErrorModelScaleFactor_ = np.sqrt( ( self.f1**4.0 + self.f5**4.0 ) / ( self.f1**2.0 + self.f5**2.0 )**2.0 )

	def gpsErrorModelSigMP_(self, elv):
		"""
		Multipath GPS user error estimate based on elevation angle
		Eq 57(2) of Blanch et. al. 2015.

		input:
			elv - satellite elevation angle from the user's perspective {deg.}
		output:
			estimate of error standard deviation {meters}
		"""
		return 0.13 + 0.53*np.exp( -elv / 10.0 )

	def gpsErrorModelSigNoise_(self, elv):
		"""
		Sigma Noise GPS user error based on elevation angle.
		Eq. 57(3) of Blanch et. al. 2015.

		input:
			elv - satellite elevation angle from the user's perspective {deg.}
		output:
			estimate of error standard deviation {meters}
		"""
		return 0.15 + 0.43*np.exp( -elv / 6.9 )

	def gpsErrorModel_(self, elv):
		"""
		GPS Error model, Equation 57(1) of Blanch et. al. 2015.

		input:
			elv - satellite elevation angle from the user's perspective {deg.}
		output:
			estimate of GPS Signal In Space User Error {meters}
		"""
		return self.gpsErrorModelScaleFactor_*np.sqrt( gpsErrorModelSigNoise_(elv)**2.0 + gpsErrorModelSigMP_(elv)**2.0 )

	def tropErrorModel_(self, elv):
		"""
		Trop Error Model model used in baseline ARAIM Blanch et. al. 2015. Eq. 58
		
		input:
			elv - satellite elevation angle from the user's perspective {deg.}
		output:
			estimate of the tropospheric delay {meters}
		"""

		return 0.12* ( 1.001 / np.sqrt( 0.002001 + ( np.sin( ( np.pi * elv ) / 180.0 ) )**2.0 ) )

	def calculatePRCovariance(self, satsXYZ, usrPos, constellationFlags ):
		"""
		Compute the pseudorange covariance matrix for integrity Cint and Cacc

		inputs:
			satsXYZ - ECEF XYZ list of satellite positions {np.array}(#Sats by 3)
			usrPos  - ECEF XYZ user positions {np.array}(1 by 3)
			constellationFlags - 'GPS' 'GLO' 'GAL' or 'BED'
		outputs:
			Cint - diagonal elements of the nominal error model used for integrity
			Cacc - diagonal elements of the nominal error model used for continuity
		"""

		self.Cint = []
		self.Cacc = []

		self.Nsat = np.shape( satsXYZ )[0]

		for i in range( self.Nsat ):
			elv = navutils.calcElAz( satsXYZ[i] , usrPos )[0]
			if contellationFlags[i] == 'GPS':
				userError = gpsUserErrorModel_( elv )
			elif constellationFlags[i] == 'GAL':
				userError = gallileoUserErrorModel_( elv )
			else:
				print " Only GPS and Galileo Are Currently Supported "

			tropErr = tropErrorModel_( elv )
			self.Cint.append( self.sigURA[i]**2.0 + tropErr**2.0 + userError**2.0 )
			self.Cacc.append( self.sigURE[i]**2.0 + tropErr**2.0 + userError**2.0 )


	
	def ISMUpdate(self, sigURA, sigURE, maxBiasNom, pSats, pConstellation):

		"""
		Update the algorithm parameters that come from the Integrity Support Message (ISM)
		these are assuming a baseline ISM from Table I in (Blanch et. al.; 2015)

		inputs:
			sigURA - std. dev. of the clock and ephemeris error for Integrity {np.array}(#Sat by 1)
			sigURE - std. dev. of the clock and ephemeris error for Accuracy/Continuity {np.array}(#Sat by 1)
			maxBiasNom - max nominal bias for sat. i used for integrity {np.array}(#Sat by 1)
			pSats - prior probability of fault in sat i per approach {np.array}(#Sat by 1)
			pConstellation - prior probability of fault affecting more than 1 sat in constellation {np.array}(#Cont by 1)

		"""
		self.sigURA = sigURA
		self.sigURE = sigURE
		self.maxBiasNim = maxBiasNom
		self.pSats = pSats
		self.pConstellation = pConstellation


	def galileoUserErrorModel_( usrPos, satPos ):

		"""
		Calculate the elevation angle from the user's perspective 
		in order to call the emperical error model
		
		inputs:
			usrPos - ECEF XYZ position of the user {meters}
			satPos - ECEF XYZ position of the satellite {meters}
		
		"""
		elv = navutils.calcElAz( ursPos, satPos )[0]
		sigUserGal = float( galileoErrorModelInerpolant_( elv ) )**2.0

	def gpsNominalSpaceUserError_( usrPos, satPos ):

		"""






		



