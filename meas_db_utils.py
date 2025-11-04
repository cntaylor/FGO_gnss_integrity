"""meas_db_utils

Lightweight utilities for accessing measurement and snapshot data from the
local SQLite schema and a small set of pure-numpy GNSS helper functions.

This module is intentionally small and readable for revision 0. For later
revisions consider splitting DB access (meas_db) and numeric models
(meas_model) into separate modules.

Expectations:
- Database schema contains tables: Snapshots, Satellite_Locations, MC_Samples,
  Measurements with the columns referenced below.
- Numpy arrays are used for locations and measurement matrices.

The functions here intentionally accept a sqlite3.Connection object named
`conn` and return plain Python / numpy types.
"""

import sqlite3
import logging
from typing import List, Tuple, Sequence, Optional, Union

import numpy as np

log = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    # Basic configuration when module used directly
    logging.basicConfig(level=logging.INFO)

consts = {
    'c': 299792458,  # speed of light in meters / second
    'omega_ecef_eci': 7.292115E-5,  # rotation of earth in radians / second
}

def get_snapshot_ids(conn: sqlite3.Connection, dataset_name: Optional[str] = None) -> List[int]:
    '''
    Returns a unique list of all Snapshot_IDs.

    Args:
        conn (sqlite3.Connection): An active database connection object.
        dataset_name (str, optional): If provided, returns Snapshot_IDs 
                                      associated only with this Dataset.
                                      If None, returns all unique Snapshot_IDs.

    Returns:
        list: A list of integers representing the unique Snapshot_IDs.
    '''
    cursor = conn.cursor()
    try:
        sql_query = "SELECT DISTINCT Snapshot_ID FROM Snapshots"
        params = []
        if dataset_name is not None:
            sql_query += " WHERE Dataset = ?"
            params.append(dataset_name)
        sql_query += " ORDER BY Snapshot_ID ASC;"
        cursor.execute(sql_query, tuple(params))
        rows = cursor.fetchall()
        return [r[0] for r in rows]
    except sqlite3.Error as e:
        log.error("An error occurred while retrieving Snapshot IDs: %s", e)
        raise
    finally:
        cursor.close()

def get_snapshot_data(conn: sqlite3.Connection, snapshot_ids: Union[int, Sequence[int]]) -> Optional[Union[Tuple[np.ndarray, np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]]:
    '''
    Retrieves truth and satellite location data for one or more Snapshot_IDs.
    
    The return structure depends on the input:
    - If a single integer Snapshot_ID is passed, returns a single tuple: 
      (truth_location_numpy_array, satellite_locations_numpy_array).
    - If a list/tuple of Snapshot_IDs is passed, returns a list of tuples
      in the same format as the single, ordered by the input Snapshot_ID list.

    Returns None if any required data is missing or if a database error occurs.
    '''
    cursor = conn.cursor()
    try:
        # Normalize input
        if isinstance(snapshot_ids, int):
            is_single_id = True
            id_list = [snapshot_ids]
        elif isinstance(snapshot_ids, (list, tuple, np.ndarray)):
            is_single_id = False
            id_list = list(snapshot_ids)
        else:
            raise TypeError('snapshot_ids must be int or sequence of ints')

        if not id_list:
            raise ValueError('snapshot_ids must be non-empty')

        unique_ids = list(set(id_list))
        placeholders = ', '.join(['?'] * len(unique_ids))

        results_map = {sid: {'truth': None, 'sats': []} for sid in unique_ids}

        # Truths
        cursor.execute(f"""
            SELECT Snapshot_ID, T.True_Loc_X, T.True_Loc_Y, T.True_Loc_Z
            FROM Snapshots T
            WHERE Snapshot_ID IN ({placeholders})
        """, unique_ids)
        for row in cursor.fetchall():
            sid = row[0]
            results_map[sid]['truth'] = np.array(row[1:])

        # Satellite locations
        cursor.execute(f"""
            SELECT S.Snapshot_ID, S.Loc_X, S.Loc_Y, S.Loc_Z
            FROM Satellite_Locations S
            WHERE Snapshot_ID IN ({placeholders})
            ORDER BY Snapshot_ID, Satellite_num ASC
        """, unique_ids)
        for row in cursor.fetchall():
            sid = row[0]
            if sid in results_map:
                results_map[sid]['sats'].append(row[1:])

        final_output = []
        for sid in id_list:
            if sid not in results_map or results_map[sid]['truth'] is None:
                raise ValueError(f'Data not found in database for Snapshot_ID {sid}')
            data = results_map[sid]
            truth_data = data['truth']
            sat_list = data['sats']
            if len(sat_list) == 0:
                raise ValueError(f'Missing satellite data for Snapshot_ID {sid}')
            sat_array = np.array(sat_list)
            final_output.append((truth_data, sat_array))

        return final_output[0] if is_single_id else final_output
    except sqlite3.Error as e:
        log.error('An error occurred accessing the database: %s', e)
        raise
    finally:
        cursor.close()

def get_mc_sample_ids(conn: sqlite3.Connection, mc_run_id: int, dataset_name: Optional[str] = None) -> List[int]:
    '''
    Return a list of all MC_Sample_IDs associated with a particular MC_run.

    Args:
        cursor (sqlite3.Cursor): An active database cursor object.
        MC_run_ID (int): The ID of the Monte Carlo run.
        dataset_name (str, optional): If provided, limits the samples to 
                                      those linked to this Dataset.

    Returns:
        list: A list of MC_Sample_IDs
    '''
    cursor = conn.cursor()
    try:
        sql_query = """
            SELECT MCS.MC_Sample_ID
            FROM MC_Samples MCS
        """
        params = [mc_run_id]
        where_clauses = ["MCS.MC_Run_ID = ?"]
        if dataset_name is not None:
            sql_query += " JOIN Snapshots S ON MCS.Snapshot_ID = S.Snapshot_ID"
            where_clauses.append("S.Dataset = ?")
            params.append(dataset_name)
        if where_clauses:
            sql_query += " WHERE " + " AND ".join(where_clauses)
        sql_query += " ORDER BY MCS.MC_Sample_ID ASC"
        list_of_tuples = cursor.execute(sql_query, tuple(params)).fetchall()
        return [t[0] for t in list_of_tuples]
    except sqlite3.Error as e:
        log.error('An error occurred while retrieving MC Sample IDs: %s', e)
        raise
    finally:
        cursor.close()

def get_mc_samples_measurements(conn: sqlite3.Connection, mc_sample_ids: Sequence[int]) -> List[np.ndarray]:
    '''
    Retrieves the complete measurement and associated satellite location data 
    for a list of MC_Sample_IDs.

    Args:
        conn (sqlite3.Connection): An active database connection object.
        mc_sample_ids (list/tuple): A list or tuple of MC_Sample_IDs.

    Returns:
        list: A list of numpy arrays, where each array corresponds to an input 
              MC_Sample_ID. Each array is Nx4: [pseudorange, sat_X, sat_Y, sat_Z].
              Returns an empty list if no IDs are provided or on database error.
    '''
    if not mc_sample_ids:
        raise ValueError('mc_sample_ids must be non-empty')

    cursor = conn.cursor()
    try:
        unique_ids = list(set(mc_sample_ids))
        placeholders = ', '.join(['?'] * len(unique_ids))
        grouped_data = {sid: [] for sid in unique_ids}

        cursor.execute(f"""
            SELECT 
                MS.MC_Sample_ID,
                MS.Pseudorange,
                L.Loc_X,
                L.Loc_Y,
                L.Loc_Z
            FROM Measurements MS
            JOIN MC_Samples M ON MS.MC_Sample_ID = M.MC_Sample_ID
            JOIN Satellite_Locations L ON 
                M.Snapshot_ID = L.Snapshot_ID AND 
                MS.Satellite_num = L.Satellite_num
            WHERE MS.MC_Sample_ID IN ({placeholders})
            ORDER BY MS.MC_Sample_ID ASC, MS.Satellite_num ASC;
        """, unique_ids)

        for row in cursor.fetchall():
            sample_id = row[0]
            data_row = row[1:]
            if sample_id in grouped_data:
                grouped_data[sample_id].append(data_row)

        final_output = []
        for input_id in mc_sample_ids:
            data_list = grouped_data.get(input_id)
            if data_list is None or len(data_list) == 0:
                final_output.append(np.empty((0, 4), dtype=float))
            else:
                final_output.append(np.array(data_list))
        return final_output
    except sqlite3.Error as e:
        log.error('An error occurred during measurement retrieval: %s', e)
        raise
    finally:
        cursor.close()

def get_mc_sample_truths(conn: sqlite3.Connection, mc_sample_ids: Sequence[int]) -> List[np.ndarray]:
    '''
    Retrieves the truth location (X, Y, Z) for each associated snapshot ID 
    for a list of MC_Sample_IDs. Throws an error if any requested ID is not found.

    Args:
        conn (sqlite3.Connection): An active database connection object.
        mc_sample_ids (list/tuple): A list or tuple of MC_Sample_IDs.

    Returns:
        list: A list of numpy arrays, where each array is a 3-element vector 
              [True_Loc_X, True_Loc_Y, True_Loc_Z] corresponding to an input ID.
              Returns an empty list if any error occurs.
              
    Raises:
        ValueError: If any MC_Sample_ID is not found in the database.
    '''
    
    if not mc_sample_ids:
        raise ValueError('mc_sample_ids must be non-empty')

    cursor = conn.cursor()
    try:
        unique_ids = list(set(mc_sample_ids))
        placeholders = ', '.join(['?'] * len(unique_ids))

        cursor.execute(f"""
            SELECT 
                M.MC_Sample_ID,
                S.True_Loc_X,
                S.True_Loc_Y,
                S.True_Loc_Z
            FROM MC_Samples M
            JOIN Snapshots S ON M.Snapshot_ID = S.Snapshot_ID
            WHERE M.MC_Sample_ID IN ({placeholders})
        """, unique_ids)

        raw_results = cursor.fetchall()
        if len(raw_results) != len(unique_ids):
            found_ids = {row[0] for row in raw_results}
            missing_ids = set(unique_ids) - found_ids
            raise ValueError(
                f"Data integrity error: The following MC_Sample_IDs were not found in the database: {list(missing_ids)}"
            )

        truth_map = {row[0]: np.array(row[1:]) for row in raw_results}
        final_output = [truth_map[input_id] for input_id in mc_sample_ids]
        return final_output
    except (sqlite3.Error, ValueError) as e:
        log.error('Error during MC sample truth retrieval: %s', e)
        raise
    finally:
        cursor.close()

def compute_noiseless_model(conn: sqlite3.Connection, snapshot_ids: Union[int, Sequence[int]]):
    '''
    For a given snapshot ID (or list of IDs), retrieve the truth location and satellite locations,
    then compute the noiseless pseudoranges for each satellite.

    Args:
        conn (sqlite3.Connection): An active database connection object.
        snapshot_ids (int or list of int): The Snapshot_ID(s) for which to compute pseudoranges.

    The return structure depends on the input:
    - If a single integer Snapshot_ID is passed, returns a single numpy array
        of noiseless pseudoranges for that snapshot.
    - If a list/tuple of Snapshot_IDs is passed, returns a list of tuples  the same format, ordered by the input Snapshot_ID list,
    Returns None if any required data is missing or if a database error occurs.

    '''
    snapshot_list = get_snapshot_data(conn, snapshot_ids)

    def pseudoranges_for_snapshot(snapshot_data):
        true_loc, satellites = snapshot_data
        pseudoranges = np.zeros(len(satellites))
        for i in range(len(satellites)):
            pseudoranges[i] = compute_noiseless_pseudorange(true_loc, satellites[i])
        return pseudoranges

    if isinstance(snapshot_ids, int):
        return pseudoranges_for_snapshot(snapshot_list)

    pseudoranges_list = []
    for snapshot_data in snapshot_list:
        pseudoranges_list.append(pseudoranges_for_snapshot(snapshot_data))
    return pseudoranges_list

def ut_l2_vs_noiseless_model(conn: sqlite3.Connection, sample_ids: Sequence[int]) -> bool:
    """Simple unit test: generated noiseless pseudoranges should recover truth via L2.

    Returns True when all tested samples are within tolerance.
    """
    all_passed = True
    try:
        created_pseudoranges = compute_noiseless_model(conn, sample_ids)
    except Exception as e:
        log.error('Failed to compute noiseless pseudoranges: %s', e)
        return False

    for i, snapshot_id in enumerate(sample_ids):
        try:
            data = get_snapshot_data(conn, snapshot_id)
        except Exception as e:
            log.error('Data retrieval failed for Snapshot_ID %s: %s', snapshot_id, e)
            all_passed = False
            continue

        true_loc, satellites = data
        measurement_array = np.zeros((len(satellites), 4))
        measurement_array[:, 0] = created_pseudoranges[i]
        measurement_array[:, 1:] = satellites

        est_loc, est_time_offset = l2_estimate_location(measurement_array)

        position_error = np.linalg.norm(est_loc - true_loc)
        if position_error > 0.001:
            log.error('Test failed for Snapshot_ID %s: Position error %s m exceeds tolerance.', snapshot_id, position_error)
            all_passed = False
        if est_time_offset > 0.1:
            log.error('Test failed for Snapshot_ID %s: Time offset %s s exceeds tolerance.', snapshot_id, est_time_offset)
            all_passed = False

    if all_passed:
        log.info('All tests passed: L2 estimation matches noiseless pseudorange model within tolerance.')
    return all_passed

def insert_mc_samples(conn: sqlite3.Connection, mc_run_id: int, data_list: Sequence[dict]) -> None:
    '''
    Adds entries to the MC_Samples and Measurements tables for a batch of data.

    For each entry in data_list:
    1. Validates the number of pseudoranges against the number of satellites in Satellite_Locations.
    2. Inserts into MC_Samples to get a new MC_Sample_ID.
    3. Inserts pseudorange measurements (and other data if in the data_list) into the Measurements table.

    Args:
        conn (sqlite3.Connection): An active database connection object.
        mc_run_id (int): The ID of the Monte Carlo run (must exist in MC_Runs).
        data_list (list): A list of dictionaries.  Each dictionary will have:
            Required:
                - 'Snapshot_ID': The ID of the snapshot (must exist in Snapshots).
                - 'pseudoranges': A 1D numpy array of pseudorange measurements.
            Optional:
                - 'is_outlier': A 1D numpy array of integers (0 or 1) indicating outliers. (Same length as pseudoranges)

    Raises:
        ValueError: If a pseudorange array size does not match the known satellite count.
        sqlite3.Error: For any database integrity or operational errors.
    '''
    
    cursor = conn.cursor()

    try:
        # --- 1. Pre-fetch Satellite Counts ---
        # Get the expected number of satellites for every unique Snapshot_ID in the input list.
        snapshot_ids = [data['Snapshot_ID'] for data in data_list]
        unique_ids = list(set(snapshot_ids))

        if not unique_ids:
            return

        placeholders = ', '.join(['?'] * len(unique_ids))

        # Query to count satellites for all relevant snapshots in one go
        cursor.execute(f"""
            SELECT Snapshot_ID, COUNT(Satellite_num)
            FROM Satellite_Locations
            WHERE Snapshot_ID IN ({placeholders})
            GROUP BY Snapshot_ID;
        """, unique_ids)

        # Map: {Snapshot_ID: count_of_satellites}
        sat_counts = dict(cursor.fetchall())

        # --- 2. Process and Insert Data Transaction by Transaction ---

        for snapshot_dict in data_list:
            # Begin a transaction for this snapshot (crucial for integrity check)
            cursor.execute("BEGIN TRANSACTION")

            snapshot_id = snapshot_dict['Snapshot_ID']
            pseudorange_array = snapshot_dict['pseudoranges']
            # A. VALIDATION CHECKS
            expected_count = sat_counts.get(snapshot_id)
            actual_count = pseudorange_array.size
            outlier_array_valid = False

            if expected_count is None:
                # This could mean the Snapshot_ID doesn't exist or has no satellites
                raise ValueError(f"Snapshot_ID {snapshot_id} not found or has no satellite data in Satellite_Locations.")

            if actual_count != expected_count:
                # Check you have the right number of satellites in the pseudorange data
                raise ValueError(
                    f"Pseudorange array for Snapshot_ID {snapshot_id} is incorrect size. "
                    f"Expected {expected_count} satellites, got {actual_count} measurements."
                )

            if 'is_outlier' in snapshot_dict:
                is_outlier_array = snapshot_dict['is_outlier']
                if is_outlier_array.size != expected_count:
                    raise ValueError(
                        f"Is_Outlier array for Snapshot_ID {snapshot_id} is incorrect size. "
                        f"Expected {expected_count}, got {is_outlier_array.size}."
                    )
                outlier_array_valid = True

            # B. INSERT INTO MC_SAMPLES
            cursor.execute("""
                INSERT INTO MC_Samples (MC_Run_ID, Snapshot_ID)
                VALUES (?, ?)
            """, (mc_run_id, snapshot_id))

            new_mc_sample_id = cursor.lastrowid

            if new_mc_sample_id is None:
                # Should not happen under normal circumstances, but good for safety
                raise sqlite3.Error("Failed to retrieve new MC_Sample_ID.")

            # C. PREPARE DATA FOR MEASUREMENTS INSERTION
            # The list of pseudoranges is assumed to be ordered by Satellite_num (0, 1, 2, ...)
            # We create a list of tuples: [(MC_Sample_ID, Satellite_num, Pseudorange), ...]

            # Assuming satellite numbers start at 0 and increment sequentially up to expected_count
            if outlier_array_valid:
                measurement_data = [
                    (new_mc_sample_id, sat_num, prange, is_outlier)
                    for sat_num, (prange, is_outlier) in enumerate(zip(pseudorange_array, is_outlier_array))
                ]
            else:
                measurement_data = [
                    (new_mc_sample_id, sat_num, prange, None)
                    for sat_num, prange in enumerate(pseudorange_array)
                ]

            # D. INSERT INTO MEASUREMENTS (using executemany for speed)
            cursor.executemany("""
                INSERT INTO Measurements (MC_Sample_ID, Satellite_num, Pseudorange, Is_Outlier)
                VALUES (?, ?, ?, ?)
            """, measurement_data)

            # Commit the single transaction after all insertions succeed
            conn.commit()

    except (sqlite3.Error, ValueError) as e:
        # If any error (database or validation) occurs, rollback the current transaction
        conn.rollback()
        raise e  # Re-raise the error so the calling function knows the insertion failed
    except Exception as e:
        # Handle unexpected Python errors
        conn.rollback()
        raise e

def l2_estimate_location(measurement_array: np.ndarray) -> Tuple[np.ndarray, float]:
    '''
    Take in an Nx4 array where N is the number of satellites for that time. 
    Columns are [pseudorange, satellite_X, satellite_Y, satellite_Z].
    XYZ assumed to be ECEF at the time the signal was broadcast

    Compute the estimated location and time-offset and return it as a 4-element np.array
    (loc.x, loc.y, loc.z, time_offset)
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

def compute_noiseless_pseudorange(true_loc: np.ndarray, sat_loc: np.ndarray) -> float:
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

if __name__ == "__main__":
    db_name = "chemnitz_data.db"
    dataset_name = 'Chemnitz'
    conn = sqlite3.connect(db_name)
    # Run the unit test for L2 and pseudorange data 
    sample_ids = get_snapshot_ids(conn, dataset_name=dataset_name)
    test_passed = ut_l2_vs_noiseless_model(conn, sample_ids)


    try:
        data = get_snapshot_data(conn, 5) # To pick a random one
    except Exception as e:
        print(f"Nothing retrieved from database: {e}")
    else:
        true_loc, satellites = data
        print('True loc of ',true_loc)
        print('Satellites:\n',satellites)
    #from Chemnitz, time = .748
    test_data = \
        np.array([  [20433081.468962, 13962834.55507, -10147463.387537, 20062447.537719],
                    [22741704.38303, 8272953.9491276, 23476859.616882, 10113690.080032],
                    [19569440.320918, 13695700.569672, 8162502.6130882, 21290888.841658],
                    [19802243.93382, 21476482.882032, -1924493.9700386, 15398448.047354],
                    [22146229.904107, 9699671.3804291, -18185112.949658, 16328149.000088],
                    [23711215.457442, 21584366.047531, -15252931.240547, -506880.06379425],
                    [23894759.19952, -11182401.657468, 11048376.816374, 21643735.158597],
                    [23287964.196421, -678474.06015606, -16622774.622281, 20828872.043502],
                    [24206377.55154, 20908080.276127, 15204464.094461, -6648871.4814316],
                    [20438005.575857, 14221365.13697, 16453041.738554, 15142841.879854],
                    [24172290.966639, 25018569.394862, 4653177.0689317, -7970367.3983922]])               
    est_loc = l2_estimate_location(test_data)
    print('Estimated',est_loc)
    truth_data = np.array([3934098.6695941, 902425.34303717, 4922416.4287205])
    print('Truth:',truth_data)
    print('Error:',est_loc[0]-truth_data)


'''
Possible test sample for add_MC_samples


# --- Example Usage (Requires prior table and data setup) ---
if __name__ == '__main__':
    # WARNING: This section is for demonstration and requires a pre-existing 
    # database file with Snapshots, Satellite_Locations, and MC_Runs (with ID=10) populated.
    
    DB_FILE = "tracking_data_option_a.db"

    # Mock Data:
    # Snapshot 101 has 5 satellites.
    # Snapshot 102 has 4 satellites.
    correct_array_101 = np.array([20000.1, 20000.2, 20000.3, 20000.4, 20000.5]) # Size 5
    correct_array_102 = np.array([30000.1, 30000.2, 30000.3, 30000.4])          # Size 4
    bad_array_101 = np.array([100.1, 100.2, 100.3]) # Size 3 (should fail validation for 101)

    # List of data to insert
    data_to_insert = [
        (101, correct_array_101),
        (102, correct_array_102),
        (101, bad_array_101), # This item is intended to fail
    ]
    
    try:
        conn = sqlite3.connect(DB_FILE)
        print("Starting insertion test...")
        # Assume MC_Run_ID 10 exists
        add_MC_samples(conn, 10, data_to_insert)
        print("Insertion complete (if no errors were printed).")
    
    except ValueError as ve:
        print(f"\n--- Validation Error Caught ---")
        print(ve)
        print("All database changes were rolled back due to this error.")
    
    except sqlite3.Error as se:
        print(f"\n--- Database Error Caught ---")
        print(se)
        
    finally:
        if 'conn' in locals() and conn:
            conn.close()
'''