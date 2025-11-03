#!/usr/bin/env python

import numpy as np

# $PNG rc script must have been sourced
import Integrity

gnssIntegrity = Integrity.KFIntegrityMonitor()
gnssIntegrity2=Integrity.KFIntegrityMonitor(probabilityOfFailure=1e-9, alertLimit=100)

H= np.matrix(' .5 ,1 ,2; 1 3 6.25; .3 16 .2')
x= np.matrix('1; 2; 3.14')
z= np.matrix('7; .6; .5')

gnssIntegrity.postfitResiduals(z,x,H)

R=np.eye(3)*2.5
gnssIntegrity.accumulateRSOS(R)


