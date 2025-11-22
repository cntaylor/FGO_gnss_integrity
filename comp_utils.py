import numpy as np
from typing import Tuple, Sequence, List

consts = {
    'c': 299792458,  # speed of light in meters / second
    'omega_ecef_eci': 7.292115E-5,  # rotation of earth in radians / second
    'earth_rotation_vec': np.array([0.0, 0.0, 7.292115E-5])  # rotation vector of earth
    'f1' :  1575.42e6, # Hz
    'f2' = 1227.60e6, # Hz
    'f5' = 1176.45e6, # Hz
}

def compute_ecef_at_current_time(sat_loc: np.ndarray, time_offset: float) -> np.ndarray:
    '''
    Given a satellite location in ECEF at the time of broadcast, compute its location
    in ECEF for the current time given by time_offset (in seconds).

    Args:
        sat_loc: A numpy array of shape (3,) representing the satellite location in ECEF coordinates.
        time_offset: A float representing the time offset in seconds (distance between location and satellite / consts['c']).

    Returns:
        A numpy array of shape (3,) representing the satellite location in ECEF coordinates at the current time.
    '''
    sat_offset = np.cross(-time_offset * consts['earth_rotation_vec'], sat_loc)
    new_sat_loc = sat_loc + sat_offset
    return new_sat_loc

def compute_pseudorange(true_loc: np.ndarray, sat_loc: np.ndarray) -> Tuple[float, np.ndarray]:
    '''
    Compute the noiseless pseudorange between a true location and a satellite location.
    Both inputs are 3-element numpy arrays representing ECEF coordinates.
    Returns the pseudorange as a float and the satellite location at the time of transmission,
    but in the ECEF frame of the time of reception.
    '''
    # Because it takes time to travel and ECEF changes
    # over time, find the sat_loc _in the receiving time_ ECEF
    rotation_vec = np.zeros(3)
    rotation_vec[2] = consts['omega_ecef_eci']

    distance = np.linalg.norm(true_loc - sat_loc)
    delta_dist = 10000.0
    while delta_dist > 0.01:
        time_traveled = distance / consts['c']
        curr_sat_loc = compute_ecef_at_current_time(sat_loc, time_traveled)
        new_distance = np.linalg.norm(true_loc - curr_sat_loc)
        delta_dist = abs(new_distance - distance)
        distance = new_distance
    return distance, curr_sat_loc

def compute_snapshot_pseudoranges(snapshot_data: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    true_loc, satellites = snapshot_data
    pseudoranges = np.array([compute_pseudorange(true_loc, s)[0] for s in satellites])
    return pseudoranges

def compute_list_snapshot_pseudoranges(snapshot_data_list: Sequence[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
    '''
    Take a list of snapshot_data (like returned from get_snapshot_data in meas_db_utils),
        then compute the noiseless pseudoranges for each satellite.

    Args:
        snapshot_data: A sequence of tuples, each containing:
            - true_loc: A numpy array of shape (3,) representing the true receiver location in ECEF coordinates.
            - satellites: A numpy array of shape (N, 3) representing the satellite locations in ECEF coordinates.

    Return:
        A list of numpy arrays, each containing the noiseless pseudoranges for the corresponding snapshot_data entry.
    '''

    pseudoranges_list = [compute_snapshot_pseudoranges(sd) for sd in snapshot_data_list]
    return pseudoranges_list

def compute_residual_and_jacobian(measured_pseudoranges: np.ndarray, 
                                  estimated_location: np.ndarray, 
                                  satellite_locations: np.ndarray, 
                                  time_offset: float,
                                  compute_Jacobian: bool = False) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
    '''
    Compute the residuals between measured pseudoranges and those computed from an estimated location.

    Args:
        measured_pseudoranges: A numpy array of shape (N,) containing the measured pseudoranges.
        estimated_location: A numpy array of shape (3,) representing the estimated receiver location in ECEF coordinates.
        satellite_locations: A numpy array of shape (N, 3) representing the satellite locations in ECEF coordinates.
        time_offset: A float representing the estimated receiver clock bias in meters.
    Returns:
        if compute_Jacobian is False, returns a numpy array of shape (N,) containing the residuals for each satellite
        else a Tuple with (residuals, Jacobian)
    '''
    residuals = np.zeros_like(measured_pseudoranges)
    sat_locs = np.zeros((len(residuals),3))
    for i in range(len(residuals)):
        pseudo_range, sat_locs[i] = compute_pseudorange(estimated_location, satellite_locations[i])
        residuals[i] = measured_pseudoranges[i] - pseudo_range - time_offset
    if not compute_Jacobian:
        return residuals
    J = np.zeros((len(residuals),4)) # columns are x,y,z of location and time_offset
    J[:,3] = 1
    for i in range(len(residuals)):
        diff = estimated_location - sat_locs[i]
        J[i,:3] = diff/np.linalg.norm(diff)
    return (residuals, J)

def estimate_l2_location (measurement_array: np.ndarray) -> Tuple[np.ndarray, float]:
    '''
    Take in an Nx4 array where N is the number of satellites for that time. 
    Columns are [pseudorange, satellite_X, satellite_Y, satellite_Z].
    XYZ assumed to be ECEF at the time the signal was broadcast

    Compute the estimated location and time-offset and return it as a tuple with
    (location (3-element np.array), time in m(float))
    '''
    n_sats = len(measurement_array)
    assert (n_sats >= 4), "Not enough locations to run L2 estimation.  Need at least 4 satellites"

    # Start the optimization procedure to find the location
    est_loc = np.zeros(3)
    est_time_offset = 0.0
    curr_time_meas = measurement_array.copy()
    mag_delta = 10000.0
    while mag_delta > 0.1:  # Iterate till changing less than 0.1 meters
        # Compute the Jacobian matrix -- unit vectors going from current location to satellite locations
        # Plus a bias for receiver clock
        y,J = compute_residual_and_jacobian(measurement_array[:,0], est_loc, 
                                            curr_time_meas[:,1:],est_time_offset, True)
        # Solve for and apply the delta
        delta = np.linalg.lstsq(J,y)[0]
        est_loc += delta[:3]
        est_time_offset += delta[3]
        mag_delta = np.linalg.norm(delta) # When small enough, no more iterations
    return (est_loc, est_time_offset)

def compute_satellite_elevation (receiver_loc: np.ndarray, sat_loc: np.ndarray) -> float:
    '''
    Compute the elevation angle between a receiver and a satellite in radians

    Args:
        receiver_loc: A numpy array of shape (3,) representing the receiver location in ECEF coordinates.
        sat_loc: A numpy array of shape (3,) representing the satellite location in ECEF coordinates.
    '''
    receiver_lat_lon = r3f.ecef_to_geodetic(receiver_loc)
    C_n_ecef = r3f.dcm_ecef_to_navigation(receiver_lat_lon[0], receiver_lat_lon[1])
    diff_ecef = sat_loc-receiver_loc
    sat_loc_n = np.dot(C_n_ecef, diff_ecef)
    return np.arctan2(sat_loc_n[2], np.linalg.norm(sat_loc_n))

def test_estimate_l2_roundtrip(snapshot_data: Tuple[np.ndarray, np.ndarray], tol=0.1):
    """Unit-test helper: For a snapshot_data tuple, build synthetic measurement array
    from truth+satellites, run estimate_l2_location and assert estimated
    location is within `tol` meters of truth.

    Args:
        snapshot_id_list: Tuple with (truth_data, measurement_data)
        tol: float tolerance in meters

    Returns:
        None, but raises AssertionError on failure.
    """
    truth, sats = snapshot_data
    pranges = compute_snapshot_pseudoranges(snapshot_data)
    # Build measurement array Nx4: [pseudorange, sat_x, sat_y, sat_z]
    meas = np.hstack([pranges.reshape(-1,1), sats])
    est_loc, est_time = estimate_l2_location(meas)
    err = np.linalg.norm(est_loc - truth)
    if err > tol:
        print(f'est_loc was {est_loc}\ntruth was {truth}')
        raise AssertionError(f'estimated location error {err:.3f} m exceeds tolerance {tol} m')
    return


if __name__ == "__main__":
    # Do a test on all to snapshots.  This unit test assumes that a database already exist
    import meas_db_utils as mdu
    import sqlite3

    conn = sqlite3.connect("meas_data.db")
    snapshot_ids = mdu.get_snapshot_ids(conn) # get all snapshots
    snapshot_data = mdu.get_snapshot_data(conn, snapshot_ids)
    for i,sd in enumerate(snapshot_data):
        if i % 100 == 0:
            print(f"On i {i}", end='\r', flush=True)
        try:
            test_estimate_l2_roundtrip(sd)
        except:
            print(f'Unit test failed for snapshot ID {snapshot_ids[i]}')
            print(f'Data was\n{snapshot_data}')
    print("Unit test completed on comp_utils")