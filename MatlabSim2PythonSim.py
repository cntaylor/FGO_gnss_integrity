#!/usr/bin/env python


import numpy as np
import scipy.linalg
import scipy.io
import os, sys

# png modules
import Transform as navutils
import Const
import utils



class GNSSData():
	"""
		Class that holds the simulated NSS Data 
	"""

	def setTrueLLH(self, latData, lonData, heightData):

		self.llh = np.matrix(np.zeros([len(latData),3]))

		self.llh[:,0] = np.array( latData )
		self.llh[:,1] = np.array( lonData )
		self.llh[:,2] = np.array( heightData )

		self.ECEFNom = navutils.llh2xyz(np.array(self.llh[0,:]).reshape(-1).tolist())

	def setAntennaOffset( self, offset):

		self.antennaOffset = np.array( offset )

	def setProduct( self, Product ):
	
		self.Product = np.array( Product )

	def setBody2Nav( self, dcm ):

		self.DcmBody2Nav = np.array( dcm )

	def setRange(self, rangeData):

		self.rangeLC = np.matrix( rangeData )

	def setPhase(self, phaseData):

		self.phaseLC = np.matrix( phaseData )

	def setSatXYZ(self, satData):

		self.satXYZ = np.array( satData )
	
	def setNumberOfSatsUsed(self, nSatData):
		
		self.nSat = np.transpose(np.array( nSatData ))
		
	
	def setSvID( self, Id ):

		self.svID = np.array( Id )

		L,n= np.shape( self.svID )
		
		for i in range(L):
			nSat = self.nSat[0,i]
			
			for j in range(nSat):
				self.svID[i,j]= self.simID2ID( self.svID[i,j] )
				

			
	def setInjectedFailures( self, injectFailData, injectFailMagData ):

		self.injectFail = np.array( injectFailData )
		self.injectFailMag = np.array( injectFailMagData ) 
	
	def carrierSmoothPR( self ):
		M = 100.0
		x,L = np.shape( self.nSat )
		self.carrierSmoothPR = np.matrix( np.zeros( np.shape( self.rangeLC ) ) )
		for i in range( L ):
			nSat = self.nSat[0,i]
			for j in range( nSat ):
				if( i == 0 ):
					self.carrierSmoothPR[i,j] = self.rangeLC[i,j]
				else:
					prn = self.svID[i,j]
					ind = utils.find( self.svID[i-1,:], lambda x: x == prn )
					if not ind:
						self.carrierSmoothPR[i,j] = self.rangeLC[i,j]
					else:
						ind = ind[0]
						self.carrierSmoothPR[i,j] = (1/M)*self.rangeLC[i,j] + \
									    ((M-1)/M)*(self.carrierSmoothPR[i-1,ind] + \
									    (self.phaseLC[i,j] - self.phaseLC[i-1,ind] ))
			
				

				
	def simID2ID(self, idNumber):

		if idNumber <= 37:
		# GPS
			return idNumber
		elif idNumber >= 51 and idNumber <= 74:
		# Glonass
			return idNumber-13
		elif idNumber >= 201 and idNumber <= 230:
		# Galileo
			return idNumber-130
		# need to add Biedou support
		else:
			return idNumber


class IMUData():

	def setDTHETA( self, gyroData ):
		
		self.DTHETA = np.array( gyroData )

	def setDVEL( self, accelData ):
		self.DVEL = np.array( accelData )

	def setTrueLLH(self, latData, lonData, heightData):

		self.llh = np.matrix(np.zeros([len(latData),3]))

		self.llh[:,0] = np.array( latData )
		self.llh[:,1] = np.array( lonData )
		self.llh[:,2] = np.array( heightData )

	def setImuSelection( self, imuSel ):

		self.ImuSelection = np.array( imuSel)

class FlightData():

	def setTime( self,Profile ):

		self.Time = np.array( Profile[:,18] )

class NoisyOrbit():

	def setIdError( self,IDError ):
		
		self.IdError = np.array( IDError )

	def setNSatsNoisy( self,nSatsNoisy ):

		self.NSatsNoisy = np.array( nSatsNoisy )

	def setSatError( self,satError ):

		self.SatError = np.array( satError )
		

	

class Simulation():
	"""
		Class that load PPP/INS matlab simulation file 

	"""
	def __init__( self, SimDir, NoisyOrbitFile=True, autoLoad = True , injectedFaults = False):
		

		self._SimDir = SimDir 
		self._injectedFaults = injectedFaults
		if( autoLoad ):
			GNSSFileName = os.path.join( SimDir, 'gpsGalReceiver1.mat' )
			if( os.path.isfile( GNSSFileName ) ):

				self.GNSSDict_ = scipy.io.loadmat( GNSSFileName )
				print " Loaded GNSS Simulation File "
			else:
			
				print " GNSS File Doesn't Exist %s "%( GNSSFileName )
				print " Exiting "
				sys.exit(1)

			IMUFileName = os.path.join( SimDir, 'GeneratedImuData.mat' )
			if( os.path.isfile( IMUFileName ) ):

				self.IMUDict_ = scipy.io.loadmat( IMUFileName )
				print " Loaded IMU Simulation File "

			else: 
				print " IMU File Doesn't Exist %s "%( IMUFileName )
				print " Exiting "
				sys.exit(1)

		
			FlightPathFileName = os.path.join( SimDir, 'GeneratedFlight.mat' )

			if( os.path.isfile( FlightPathFileName ) ):

				self.FlightPathDict_ = scipy.io.loadmat( FlightPathFileName )
				print " Loaded Flight Path Simulation File ( truth )"

			else:
				print " Flight Path File Doesn't Exist %s "%( FlightPathFileName )
				print " Exiting "
				sys.exit(1)

			NoisyOrbitName = os.path.join( SimDir, 'NoisyOrbits1.mat' )

			if( os.path.isfile( NoisyOrbitName )  ):

				self.NoisyOrbitDict_ = scipy.io.loadmat( NoisyOrbitName )
				print " Loaded Noisy Orbit File "

			elif NoisyOrbitFile:

				print " Noisy Orbit File Doesn't Exist %s "%( NoisyOrbitName )
				print " Exiting "
				sys.exit(1)



			self.parseGNSS_()
			self.parseIMU_()
			self.parseFlight_()
			if NoisyOrbitFile:
				self.parseNoisyOrbits_()

	
	def loadGNSS( self, injectedFaults = False ):
		"""
			Method to parse GNSS data. Used when auto-loading is turned off.
		"""
		self._injectedFaults = injectedFaults
		GNSSFileName = os.path.join( self._SimDir, 'gnssFaults.mat' )
		if( os.path.isfile( GNSSFileName ) ):
                	self.GNSSDict_ = scipy.io.loadmat( GNSSFileName )
                        print " Loaded GNSS Simulation File "
                else:

                       	print " GNSS File Doesn't Exist %s "%( GNSSFileName )
                        print " Exiting "
                        sys.exit(1)

		self.parseGNSS_()

	def loadIMU( self ):
		"""
			Method to parse IMU data. Used when auto-loading is turned off.
		"""
		IMUFileName = os.path.join( self._SimDir, 'GeneratedImuData.mat' )
                if( os.path.isfile( IMUFileName ) ):

                	self.IMUDict_ = scipy.io.loadmat( IMUFileName )
                        print " Loaded IMU Simulation File "

                else:
                        print " IMU File Doesn't Exist %s "%( IMUFileName )
                        print " Exiting "
                        sys.exit(1)

		self.parseIMU_()

	def parseGNSS_( self ):

		"""
			Parse the GNSS data file 

		"""
		self.GNSS = GNSSData()

		self.GNSS.setNumberOfSatsUsed( self.GNSSDict_['NumerOfSatsUsed'] )
		self.GNSS.setRange( self.GNSSDict_['RangeLC'] )
		self.GNSS.setPhase( self.GNSSDict_['PhaseLC'] )
		self.GNSS.setSatXYZ( self.GNSSDict_['Sats'] )
		self.GNSS.setSvID( self.GNSSDict_['Id'] )
		self.GNSS.setTrueLLH( self.IMUDict_['TrueLat'], \
				self.IMUDict_['TrueLon'], \
				self.IMUDict_['TrueHeight'] )
		self.GNSS.carrierSmoothPR()
		self.GNSS.setBody2Nav(self.GNSSDict_['DcmBodyToNav'] )
		self.GNSS.setAntennaOffset( self.GNSSDict_['antennaOffset'] )
		self.GNSS.setProduct( self.GNSSDict_['Product'] ) 
		if( self._injectedFaults ):
			self.GNSS.setInjectedFailures( self.GNSSDict_['InjectFail'], self.GNSSDict_['FailMag'] )

	def parseIMU_( self ):

		"""
			Parse the IMU data file
		"""
		self.IMU = IMUData()

		self.IMU.setDTHETA( self.IMUDict_['DeltaThetaNoisy'] )
		self.IMU.setDVEL( self.IMUDict_['DeltaVelNoisy'] )
		self.IMU.setTrueLLH( self.IMUDict_['TrueLat'], \
				self.IMUDict_['TrueLon'], \
				self.IMUDict_['TrueHeight'] )
		self.IMU.setImuSelection( self.IMUDict_['ImuSelection'] )

	def parseFlight_( self ):

		"""
			Parse the flight path data file 
		"""
		self.Flight = FlightData()
	
		self.Flight.setTime( self.FlightPathDict_['Profile'] )

	def parseNoisyOrbits_( self ):

		"""
			Parse the noisy orbit data file
		"""

		self.NoisyOrbits = NoisyOrbit()

		self.NoisyOrbits.setNSatsNoisy( self.NoisyOrbitDict_['nSatsNoisy'] )
		self.NoisyOrbits.setIdError( self.NoisyOrbitDict_['IDerror'] )
		self.NoisyOrbits.setSatError( self.NoisyOrbitDict_['satsError'] )

