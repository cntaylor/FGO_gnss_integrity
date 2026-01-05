import numpy as np
import comp_utils as cu
from typing import Sequence, Tuple, List
from math import sin, sqrt, pi
import scipy.interpolate
from ARAIM.IntegrityRAIMandWLS2 import MultiHypothesisSolutionSeperation

# ARAIM has a couple of different portions.  The first part is basically a way to estimate
# covariance for each satellite.  The second part is how to estimate location and protection
# level, including excluding certain faults.

def trop_error_model(elevation_angle: float) -> float:
    return 0.12* ( 1.001 / sqrt( 0.002001 + ( sin( ( elevation_angle ) ) )**2.0 ) )

def gps_user_error_model(elevation_angle: float) -> float:
    elv_degree = elevation_angle * 180 / np.pi
    # first part is regular, second for multipath...
    sig_noise = 0.15 + 0.43*np.exp( -elv_degree / 6.9 )
    mp_noise = 0.13 + 0.53*np.exp( -elv_degree / 10.0 )
    scale_factor = sqrt( cu.consts['f1']**4.0 + cu.consts['f2']**4.0 ) / ( cu.consts['f1']**2.0 + cu.consts['f2']**2.0 )**2.0
    return scale_factor * sqrt(sig_noise**2.0 + mp_noise**2.0)


galileo_elevation_angles = np.arange(0,95,5)
galileo_user_error = np.array([0.4529, 0.4529, 0.3553, 0.3063, 0.2638, 0.2593, 0.2555, 0.2504, 
			0.2438, 0.2396, 0.2359, 0.2339, 0.2302, 0.2295, 
			0.2278, 0.2297, 0.2310, 0.2274, 0.2277] )
galileo_error_interpolant = scipy.interpolate.interp1d(galileo_elevation_angles, galileo_user_error, kind='cubic')
def galileo_user_error_model(elevation_angle: float) -> float:
    return galileo_error_interpolant(elevation_angle*180.0/pi)

# sigURA should be set externally.  This is a placeholder
sig_ura_data = np.zeros(107)  

def compute_model_covariance(receiver_loc: np.ndarray, satellites_ecef: np.ndarray, svIDs: Sequence[int]) -> np.ndarray:
    '''
    Take in a list of satellite IDs, with their corresponding ecef locations, and compute the model covariance matrix

    Args:
        receiver_loc: A numpy array of shape (3,) representing the receiver location in ECEF coordinates.
        satellites_ecef: A numpy array of shape (N, 3) representing the satellite locations in ECEF coordinates.
        svIDs: A sequence of N integers representing the IDs of the satellites. 
            1 - 37 is GPS and corresponds to PRN
            38 - 61 is GLONASS and is slot + 37
            71 - 106 is GALILEO anf is PRN + 70
            62 used for unknown GLONASS
    
    Returns:
        A numpy array of shape (N, ) representing the covariance for each satellite
    '''
    assert len(satellites_ecef) == len(svIDs)

    cov = np.zeros(len(satellites_ecef))
    for i in range(len(satellites_ecef)):
        # compute the elevation of the satellite w.r.t the navigation frame at the receiver
        el = cu.compute_satellite_elevation(receiver_loc, satellites_ecef[i])
        # first, compute the User error value (ue)
        if svIDs[i] <= 37:
            ue = gps_user_error_model(el)
        elif svIDs[i] >= 71 and svIDs[i] <= 106:
            ue = galileo_user_error_model(el)
        else:
            print("Only GPS and Galileo are currently supported")

        # next, compute the tropospheric error value (te)
        te = trop_error_model(el)
        cov[i] = ue**2 + te**2 + sig_ura_data[i]**2
    return cov

ARAIM_class = None

def init_ARAIM (params : dict) -> None:
    global ARAIM_class
    if ARAIM_class is None:
        ARAIM_class = MultiHypothesisSolutionSeperation(printer=False)
    if params.get("araim_set_covariance", True): # Default is True because snapshot_ARAIM() doesn't have satellite IDs
        ARAIM_class._simpleCovMode = True
        ARAIM_class._Cint_simple = params.get("base_sigma", 10.)**2
    if params.get("araim_set_fault_prob", False):
        ARAIM_class._simpleFaultMode = True
        ARAIM_class._fault_prob = params.get("fault_prob", 0.001)
    if params.get("araim_use_bias_for_PL", False):
        ARAIM_class._useBiasForPL = False # Default is true
    if type(params.get("max_bias", False)) is float:
        ARAIM_class._maxBiasNom = [params.get("max_bias", 0.1)] * 110 # max num_sats?
    ARAIM_class._satList = list(np.arange(1,111))
    # Do I need to do a "ISM update" for things to work? It would also set URE and RUA values, 
    # plus pSats and pConstellation  Avoided if in simple mode?

def single_epoch_ARAIM (measurements: np.ndarray
                  ) -> Tuple[np.ndarray, float|None, List[int], Tuple[float,float]]:
    '''
    Run ARAIM on a snapshot of measurements

    Returns:
        est_location: A numpy array of shape (3,) representing the estimated location in ECEF coordinates.
        time_offset: A float representing the estimated time offset in meters.
        outlier_info: A list of integers representing the IDs of the satellites excluded in the estimation.
        PLs: A tuple of floats representing the HPL and VPL calculated by ARAIM
    '''
    global ARAIM_class
    if ARAIM_class is None:
        print('To use snapshot_ARAIM, should really run initialization first')
        print('with the parameters that you want.  Doing some default parameters for now.')
        init_ARAIM()
    # Run WLS estimation 
    est_location,time_offset = cu.estimate_l2_location(measurements)
    # Set value for usrLoc inside ARAIM_class
    ARAIM_class.initPos(usrPos = est_location, 
                        clockNom = time_offset)
    # Get set of new ECEF values for satellites
    new_sat_locs = np.zeros(( len(measurements), 3 ))
    for i in range(len(measurements)):
        time_offset = \
            np.linalg.norm(measurements[i][1:] - est_location) \
                / cu.consts['c'] 
        new_sat_locs[i] = cu.compute_ecef_at_current_time(measurements[i][1:], time_offset)
    # Run ARAIM.  I have no idea of the SV numbers, so random things get passed in
    HPL, VPL, outlier_info, normals_solved =ARAIM_class.ARAIM(len(new_sat_locs), new_sat_locs,
                      measurements[:,0], 
                      np.arange(1,len(measurements)+1))
    return ARAIM_class.getUpdatedPos(), None, outlier_info, (HPL, VPL), normals_solved