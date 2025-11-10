#!/usr/bin/env python


#Last Edited 7/23/2015 by Ryan


import numpy as np
import math as m
import Const

def dop(satXYZ, usrPos):
    G = hmat(satXYZ, usrPos)
    H = (G.T*G).I
    PDOP = m.sqrt(H[0,0] + H[1,1] + H[2,2])
    return (PDOP)

def calcElAz(satXYZ, staXYZ):
	enu=xyz2enu(satXYZ,staXYZ)
	mag=np.linalg.norm(enu)
	El=np.arctan2(enu[2],mag)
	Az=np.arctan2(enu[0]/mag, enu[1]/mag)
	return (El,Az)
	
def calcElAzBody(satXYZ, usrXYZ):
    enuLOS=xyz2enu(satXYZ,usrXYZ)
    R = [satXYZ - usrXYZ]
    R = np.matrix(R)
    Rt = R.T
    llh = xyz2llh(usrXYZ)
    Cen = earth2Nav(llh[0],llh[1])
    DCM = [[1,0,0],\
            [0,1,0],\
            [0,0,1]]
    bodyLOS = DCM*Cen*Rt
    El = np.arctan(-bodyLOS[2]/(np.linalg.norm([bodyLOS[0],bodyLOS[1]])))
    Az = np.arctan2(enuLOS[0],enuLOS[1])
    return (El,Az)

def tropMap(usrXYZ,satXYZ):

	El,Az=calcElAz(satXYZ,usrXYZ)
	m=1.001/np.sqrt(.002001+np.sin(El)**2);
	return m

def hmat(satPos,usrPos):
	
	N = len(satPos)
	m,n=usrPos.shape
	if m>n:
		usrPos = np.transpose(usrPos)

	h = np.ones(shape=(N,4))
	
	for i in range(N):
		tmpvec = satPos[i,:]-usrPos
		h[i,0:3]=tmpvec/np.norm(tmpvec)

	return h
	
def degMinSec2Deg(lat,lon,height):

	latDeg = lat[0] + (lat[1]/60) + (lat[2]/3600)
	lonDeg = lon[0] + (lon[1]/60) + (lon[2]/3600)

	return [latDeg, lonDeg, height]

def xyz2llh(xyz,planet=Const.Earth()):
	x2=xyz[0]*xyz[0]
	y2=xyz[1]*xyz[1]
	z2=xyz[2]*xyz[2]
	r=np.sqrt(x2+y2)
	r2=r*r
	F=54.0*planet.bSqr*z2
	G=r2+(1-planet.eSqr)*z2-planet.eSqr*planet.ESqr
	c=(planet.ESqr*planet.ESqr*F*r2)/(G*G*G)
	s=(1+c+np.sqrt(c*c+2*c))**(1/3)
	P= F/(3* ((s+1/s+1)**2)* G*G)
	Q= np.sqrt(1+2*planet.eSqr*planet.eSqr*P)
	ro= -(P*planet.eSqr*r)/(1+Q) +np.sqrt((planet.a*planet.a/2)*(1+1/Q)-(P*(1-planet.eSqr)*z2)/(Q*(1+Q))-P*r2/2)
	tmp=(r-planet.eSqr*ro)**2
	U=np.sqrt( tmp + z2 )
	V=np.sqrt( tmp + (1-planet.eSqr)*z2 )
	zo=(planet.bSqr*xyz[2])/(planet.a*V)

	height = U*(1-planet.bSqr/(planet.a*V))
	lat = np.arctan( ( xyz[2]+ planet.ep*planet.ep*zo)/r);
	temp = np.arctan(xyz[1]/xyz[0])
	if xyz[0] >= 0:
		long = temp
	elif (xyz[0] <0 ) and ( xyz[1] >=0 ):
		long = np.pi + temp
	else:
		long = temp - np.pi
	 
	return [lat,long,height]
	
def llh2xyz( llh, planet=Const.Earth() ):

	phi = llh[0]	
	lamb = llh[1]
	h = llh[2]
	a = planet.a
	b = planet.b
	e = planet.e

	sphi = np.sin(phi)
	cphi = np.cos(phi)
	clamb = np.cos(lamb)
	slamb = np.sin(lamb)
	tan2phi = np.tan( phi )**2.0
	oneMinusE2 = 1 - e*e
	denom = np.sqrt( 1 + oneMinusE2* tan2phi )

	x = (a*clamb)/denom + h*clamb*cphi
	y= (a*slamb)/denom + h*slamb*cphi
	denom2 = np.sqrt(1.0 - e*e*sphi*sphi)
	z= (a*oneMinusE2*sphi)/denom2 + h*sphi

	return [x,y,z]

def xyz2enu(xyz,org):
	orgllh=xyz2llh(org)
	diff= np.array(xyz)-np.array(org)
	diff=np.array(diff)
	phi=orgllh[0]
	lam=orgllh[1]
	sphi=np.sin(phi)
	cphi=np.cos(phi)
	slam=np.sin(lam)
	clam=np.cos(lam)
	R=np.mat( [[-slam, clam, 0],[ -sphi*clam, -sphi*slam, cphi],[ cphi*clam, cphi*slam, sphi]])
	sol=np.dot(R,diff.T)
	sol=sol.getA1()
	return sol.tolist()

def enu2xyz(enu,org):
	orgllh = xyz2llh(org)
	phi=orgllh[0]
	lam=orgllh[1]
	sphi=np.sin(phi)
	cphi=np.cos(phi)
	slam=np.sin(lam)
	clam=np.cos(lam)
	R=np.mat( [[-slam, clam, 0],[ -sphi*clam, -sphi*slam, cphi],[ cphi*clam, cphi*slam, sphi]])
	Rinv=np.linalg.inv(R)
	diffXYZ=np.dot(Rinv,enu)
	sol= np.add(org, diffXYZ)
	sol=sol.getA1()
	return sol.tolist() 

def earth2Inertial(t,to=0,omega=Const.Earth().RotationRate):

	""" Earth to Inertial Transformation 

	Inputs:
	-t [sec]--> time of transformation
	-to [sec] --> epoch the ECI and ECEF are coincident. 0 by defalt 
	-omega [rad/sec] --> rotation rate of the earth. Const.Earth().RotationRate by default """

	c = np.cos(omega*(t-to))
	s = np.sin(omega*(t-to))

	C = np.mat( [[c,s,0],\
		     [s,c,c],\
	             [0,0,1]] )
	return C


def earth2Nav(lat,lon):

	"""Earth to Nav Transformation 

	Inputs:
	-latitude [rads]
	-longitude [rads]"""
	
	slon = np.sin(lon)
	slat = np.sin(lat)
	clon = np.cos(lon)
	clat = np.cos(lat)
	C = np.mat( [[-slat*clon,-slat*slon,clat],\
		     [-slon,clon,0],\
		     [-clat*clon,-clat*slon,-slat]] )
	return C


def ENU():

	""" NED to ENU Transformation """

	C = np.mat( [[0,1,0],\
		     [1,0,0],\
		     [0,0,-1]] )

	return C


def dcm2Quat(DCMbn):     

	""" Dcm to Quaternion Transformation

	Inputs:
	-DCMbn [3x3] matrix --> Body to Nav DCM"""
	
	a = 0.5*m.sqrt(1+np.trace(DCMbn))
	sf = 1/(4*a)
	b = sf*(DCMbn[2,1] - DCMbn[1,2])
	c = sf*(DCMbn[0,2] - DCMbn[2,0])
	d = sf*(DCMbn[1,0] - DCMbn[0,1])

	return [a,b,c,d]


def dcm2Eulr(DCMbn):

	""" Dcm to Euler Angles Transformation

	Inputs:
	-DCMbn [3x3] matrix --> Body to Nav DCM

	Outputs
	-Output[1]--> Roll
	-Output[2] --> Pitch
	-Output[3] --> Yaw """

	phi = m.atan2(DCMbn[2,1],DCMbn[2,2])
	theta = m.asin(-DCMbn[2,0])
	psi = m.atan2(DCMbn[1,0],DCMbn[0,0])

	return [phi,theta,psi]


def quat2Dcm(Q):

	""" Quaternion to DCM Transformation

	Inputs:
	-Q [4x1] vector --> quaternion

	Outputs:
	-DCMbn [3x3] matrix --> Body to Nav DCM """

	A = Q[0]
	B = Q[1]
	C = Q[2]
	D = Q[3]

	DCM = np.zeros((3,3))
	DCM[0,0] = A*A + B*B - C*C -D*D
	DCM[0,1] = 2*(B*C - A*D)
	DCM[0,2] = 2*(B*D + A*C)
	DCM[1,0] = 2*(B*C + A*D)
	DCM[1,1] = A*A - B*B + C*C -D*D
	DCM[1,2] = 2*(C*D - A*B)
	DCM[2,0] = 2*(B*D - A*C)
	DCM[2,1] = 2*(C*D + A*B)
	DCM[2,2] = A*A - B*B - C*C + D*D

	return DCM



def radicurv(lat):

	e2 = Const.Earth().e*Const.Earth().e
	den = 1-e2*((np.sin(lat))*(np.sin(lat)))
	Rm = (Const.Earth().a*(1-e2))/( (den)**(3/2) )
	Rp = Const.Earth().a/( np.sqrt(den) )

	return Rm,Rp

def gravity(lat,height):

	Rm,Rp = radicurv(lat)
	Ro = np.sqrt(Rp*Rm);   # mean earth radius of curvature
	g0 = 9.780318*( 1 + 5.3024e-3*(np.sin(lat))**2 - 5.9e-6*(np.sin(2*lat))**2 )
	if height >= 0:
		g = g0/( (1 + height/Ro)**2 )
	if height < 0:
		g = g0*(1 + h/Ro)
	return g



	
