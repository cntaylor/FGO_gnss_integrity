import numpy as np
from typing import Tuple, Sequence, List

consts = {
    'c': 299792458,  # speed of light in meters / second
    'omega_ecef_eci': 7.292115E-5,  # rotation of earth in radians / second
}

def noiseless_pseudorange(true_loc: np.ndarray, sat_loc: np.ndarray) -> float:
    '''
    Compute the noiseless pseudorange between a true location and a satellite location.
    Both inputs are 3-element numpy arrays representing ECEF coordinates.
    Returns the pseudorange as a float.
    '''
    # Because it takes time to travel and ECEF changes
    # over time, find the sat_loc _in the receiving time_ ECEF
    rotation_vec = np.zeros(3)
    rotation_vec[2] = consts['omega_ecef_eci']

    distance = np.linalg.norm(true_loc - sat_loc)
    delta_dist = 10000.0
    while delta_dist > 0.01:
        time_traveled = distance / consts['c']
        sat_offset = np.cross(-time_traveled * rotation_vec, sat_loc)
        curr_sat_loc = sat_loc + sat_offset
        new_distance = np.linalg.norm(true_loc - curr_sat_loc)
        delta_dist = abs(new_distance - distance)
        distance = new_distance
    return distance

def l2_estimate_location(measurement_array: np.ndarray) -> Tuple[np.ndarray, float]:
    '''
    Take in an Nx4 array where N is the number of satellites for that time. 
    Columns are [pseudorange, satellite_X, satellite_Y, satellite_Z].
    XYZ assumed to be ECEF at the time the signal was broadcast

    Compute the estimated location and time-offset and return it as a tuple with
    (location (3-element np.array), time in m(float))
    '''
    n_sats = len(measurement_array)
    assert (n_sats >= 4), "Not enough locations to run L2 estimation.  Need at least 4 satellites"

    rotation_vec = np.zeros(3)
    rotation_vec[2] = consts['omega_ecef_eci']

    # Start the optimization procedure to find the location
    est_loc = np.zeros(3)
    est_time_offset = 0.0
    curr_time_meas = measurement_array.copy()
    mag_delta = 10000.0
    while mag_delta > 0.1:  # Iterate till changing less than 0.1 meters
        # Compute the Jacobian matrix -- unit vectors going from current location to satellite locations
        # Plus a bias for receiver clock
        J = np.zeros((n_sats,4))
        J[:,3] = 1 # clock bias Jacobian
        # and the residual vector that is being minimized (y)
        y = np.zeros(n_sats)
        for i in range(n_sats):
            diff = est_loc - curr_time_meas[i,1:]
            J[i,:3] = diff / np.linalg.norm(diff)
            y[i] = curr_time_meas[i,0] - np.linalg.norm(diff) - est_time_offset
        # Solve for and apply the delta
        delta = np.linalg.lstsq(J,y)[0]
        est_loc += delta[:3]
        est_time_offset += delta[3]
        # print('est_loc is ',est_loc, 'time is',est_time_offset, 'delta is',delta)
        # print('y is',y)
        # Change the satellite positions from when they were broadcast to when they were received
        for i in range(n_sats):
            orig_loc = measurement_array[i,1:]
            dist_traveled = np.linalg.norm(est_loc - orig_loc)  # + est_time_offset
            time_traveled = dist_traveled / consts['c']
            # Rotate the satellite position backwards by the time traveled
            sat_offset = np.cross(-time_traveled * rotation_vec, orig_loc)
            curr_time_meas[i,1:] = orig_loc + sat_offset
        mag_delta = np.linalg.norm(delta) # When small enough, no more iterations
    return (est_loc, est_time_offset)

def noiseless_model(snapshot_data_list: Sequence[Tuple[np.ndarray, np.ndarray]]) -> List[np.ndarray]:
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


    def pseudoranges_for_snapshot(snapshot_data):
        true_loc, satellites = snapshot_data
        pseudoranges = np.zeros(len(satellites))
        for i in range(len(satellites)):
            pseudoranges[i] = noiseless_pseudorange(true_loc, satellites[i])
        return pseudoranges

    pseudoranges_list = []
    for snapshot_data in snapshot_data_list:
        pseudoranges_list.append(pseudoranges_for_snapshot(snapshot_data))
    return pseudoranges_list

# def ut_l2_vs_noiseless_model(conn: sqlite3.Connection, sample_ids: Sequence[int]) -> bool:
#     """Simple unit test: generated noiseless pseudoranges should recover truth via L2.

#     Returns True when all tested samples are within tolerance.
#     """
#     all_passed = True
#     try:
#         created_pseudoranges = compute_noiseless_model(conn, sample_ids)
#     except Exception as e:
#         log.error('Failed to compute noiseless pseudoranges: %s', e)
#         return False

#     for i, snapshot_id in enumerate(sample_ids):
#         try:
#             data = get_snapshot_data(conn, snapshot_id)
#         except Exception as e:
#             log.error('Data retrieval failed for Snapshot_ID %s: %s', snapshot_id, e)
#             all_passed = False
#             continue

#         true_loc, satellites = data
#         measurement_array = np.zeros((len(satellites), 4))
#         measurement_array[:, 0] = created_pseudoranges[i]
#         measurement_array[:, 1:] = satellites

#         est_loc, est_time_offset = l2_estimate_location(measurement_array)

#         position_error = np.linalg.norm(est_loc - true_loc)
#         if position_error > 0.001:
#             log.error('Test failed for Snapshot_ID %s: Position error %s m exceeds tolerance.', snapshot_id, position_error)
#             all_passed = False
#         if est_time_offset > 0.1:
#             log.error('Test failed for Snapshot_ID %s: Time offset %s s exceeds tolerance.', snapshot_id, est_time_offset)
#             all_passed = False

#     if all_passed:
#         log.info('All tests passed: L2 estimation matches noiseless pseudorange model within tolerance.')
#     return all_passed

