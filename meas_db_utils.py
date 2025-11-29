"""meas_db_utils

Lightweight utilities for accessing measurement and snapshot data from the
local SQLite schema and a small set of pure-numpy GNSS helper functions.

There is a split for all database functions (this file) and all computation
utilities (comp_utils.py)

Expectations:
- Database schema contains tables: Snapshots, Satellite_Locations, MC_Samples,
  Measurements with the columns referenced below.
- Numpy arrays are used for locations and measurement matrices.

The functions here intentionally accept a sqlite3.Connection object named
`conn` and return plain Python / numpy types.

This library (except for the `create_measurement_database` and `add_real_data` functions)
put outputs to a logger and throws exceptions to make it as library-like as possible.
"""

import sqlite3
import logging
from typing import List, Tuple, Sequence, Optional
import json

import numpy as np

log = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    # Basic configuration when module used directly
    logging.basicConfig(level=logging.INFO)

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

def get_snapshot_data(conn: sqlite3.Connection, snapshot_ids: Sequence[int]) -> List[Tuple[np.ndarray, np.ndarray]]:
    '''
    Retrieves truth and satellite location data for one or more Snapshot_IDs.
    
    Args:
        conn (sqlite3.Connection): An active database connection object.
        snapshot_ids (list/tuple of int): The Snapshot_IDs for which to retrieve data.

    Returns a list of tuples, where each tuple is: 
      (truth_location_numpy_array, satellite_locations_numpy_array).
      - truth_location_numpy_array is a 3-element vector [True_Loc_X, True_Loc_Y, True_Loc_Z].
      - satellite_locations_numpy_array is an Nx3 array of satellite locations.
    '''
    cursor = conn.cursor()
    try:
        # Normalize input
        unique_ids = list(set(snapshot_ids))
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
        for sid in snapshot_ids:
            if sid not in results_map or results_map[sid]['truth'] is None:
                raise ValueError(f'Data not found in database for Snapshot_ID {sid}')
            data = results_map[sid]
            truth_data = data['truth']
            sat_list = data['sats']
            if len(sat_list) == 0:
                raise ValueError(f'Missing satellite data for Snapshot_ID {sid}')
            sat_array = np.array(sat_list)
            final_output.append((truth_data, sat_array))

        return final_output
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

def get_mc_samples_outliers(conn: sqlite3.Connection, mc_sample_ids: Sequence[int]) -> List[np.ndarray]:
    '''
    Retrieves the complete outlier information for a list of MC_Sample_IDs.

    Args:
        conn (sqlite3.Connection): An active database connection object.
        mc_sample_ids (list/tuple): A list or tuple of MC_Sample_IDs.

    Returns:
        list: A list of numpy arrays, where each array corresponds to a specific sample
              MC_Sample_ID. Each array is N of dtype bool (N = # pseudoranges)
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
                MS.Is_Outlier
            FROM Measurements MS
            WHERE MS.MC_Sample_ID IN ({placeholders})
            ORDER BY MS.MC_Sample_ID ASC, MS.Satellite_num ASC;
        """, unique_ids)

        for row in cursor.fetchall():
            sample_id = row[0]
            outlier_info = row[1]
            if sample_id in grouped_data:
                grouped_data[sample_id].append(outlier_info)

        final_output = []
        for input_id in mc_sample_ids:
            data_list = grouped_data.get(input_id)
            if data_list is None or len(data_list) == 0:
                final_output.append([])
            else:
                final_output.append(np.array(data_list,dtype=bool))
        return final_output
    except sqlite3.Error as e:
        log.error('An error occurred during outlier information retrieval: %s', e)
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

def get_dataset_names(conn: sqlite3.Connection) -> List[str]:
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT Dataset FROM Snapshots")
        return [row[0] for row in cursor.fetchall()]
    except sqlite3.Error as e:
        log.error('An error occurred during dataset name retrieval: %s', e)
        raise
    finally:
        cursor.close()

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
                    (new_mc_sample_id, sat_num, float(prange), bool(is_outlier))
                    for sat_num, (prange, is_outlier) in enumerate(zip(pseudorange_array, is_outlier_array))
                ]
            else:
                measurement_data = [
                    (new_mc_sample_id, sat_num, float(prange), None)
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

def create_measurement_database(db_name="measurement_data.db"):
    """
    Creates an SQLite database file with the five required tables.
    """
    try:
        # 1. Connect to (or create) the SQLite database file
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        print(f"Database '{db_name}' connected successfully.")

        # --- 1. SNAPSHOT TABLE ---
        # Stores true location information for the main object at each time step.
        # Primary Key is the snapshot.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Snapshots (
                Snapshot_ID         INTEGER PRIMARY KEY AUTOINCREMENT,
                Dataset             TEXT    NOT NULL,
                Time                REAL    NOT NULL,
                True_Loc_X          REAL    NOT NULL,
                True_Loc_Y          REAL    NOT NULL,
                True_Loc_Z          REAL    NOT NULL
            );
        """)
        print("Table 'Snapshots' created.")

        # --- 2. SATELLITE LOCATION TABLE ---
        # Stores the location of each satellite for a given Snapshot.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Satellite_Locations (
                Snapshot_ID     INTEGER NOT NULL,
                Satellite_num   INTEGER NOT NULL,
                Loc_X           REAL    NOT NULL,
                Loc_Y           REAL    NOT NULL,
                Loc_Z           REAL    NOT NULL,
                PRIMARY KEY (Snapshot_ID, Satellite_num),
                FOREIGN KEY (Snapshot_ID) 
                    REFERENCES Snapshots (Snapshot_ID)
            );
        """)
        print("Table 'Satellite_Locations' created.")
        
        # --- 3. MONTE CARLO RUN TABLE ---
        # Keeps track of all the "batches" created.  Each entry defines a "set" of samples
        # (all generated the same way.)  Records the parameters used to generate measurements 
        # (i.e., noise model).  
        # We'll use a single unique ID column. The first row (ID=0) will be 'real data'.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS MC_Runs (
                MC_Run_ID     INTEGER PRIMARY KEY AUTOINCREMENT,
                Description     TEXT NOT NULL,
                Parameters_JSON TEXT -- JSON string of the parameters
            );
        """)
        print("Table 'MC_Runs' created.")
        
        # Add a placeholder for "real data" (no modification)
        real_data_params = json.dumps({"simulated": False})
        cursor.execute("""
            INSERT INTO MC_Runs (Description, Parameters_JSON)
            VALUES (?, ?)
        """, ('Real Data (No Monte Carlo Simulation)', real_data_params))
        
        # --- 4. MONTE CARLO SAMPLES TABLE (The link table) ---
        # Links a specific timestep/snapshot to the MC run used to generate measurements.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS MC_Samples (
                MC_Sample_ID    INTEGER PRIMARY KEY AUTOINCREMENT,
                MC_Run_ID       INTEGER NOT NULL,
                Snapshot_ID     INTEGER NOT NULL,

                -- FK to ensure the time step exists
                FOREIGN KEY (Snapshot_ID)
                    REFERENCES Snapshots (Snapshot_ID),
                
                -- FK to ensure the parameters exist
                FOREIGN KEY (MC_Run_ID)
                    REFERENCES MC_Runs (MC_Run_ID)
            );
        """)
        print("Table 'MC_Samples' created.")

        # --- 5. MEASUREMENTS TABLE ---
        # Stores the generated pseudorange measurements for each Monte Carlo run.
        # Composite key ensures a unique entry for each satellite within a run.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Measurements (
                MC_Sample_ID    INTEGER NOT NULL,
                Satellite_num   INTEGER NOT NULL,
                Pseudorange     REAL    NOT NULL,
                Is_Outlier      INTEGER NULL,

                PRIMARY KEY (MC_Sample_ID, Satellite_num),
                
                -- FK to ensure the MC Run exists
                FOREIGN KEY (MC_Sample_ID)
                    REFERENCES MC_Samples (MC_Sample_ID)

            );
        """)
        print("Table 'Measurements' created.")

        # Enforce that the MC_sample_ID has the Satellite number being put into Measurements
        cursor.execute("""
            -- This syntax is specific to SQLite
            CREATE TRIGGER enforce_satellite_measurement_integrity
            BEFORE INSERT ON Measurements
            FOR EACH ROW
            BEGIN
                -- Look up the Snapshot_ID from the MC_Samples table
                SELECT CASE
                    WHEN NOT EXISTS (
                        SELECT 1 
                        FROM Satellite_Locations AS SL
                        JOIN MC_Samples AS MS ON MS.Snapshot_ID = SL.Snapshot_ID
                        WHERE 
                            MS.MC_Sample_ID = NEW.MC_Sample_ID AND 
                            SL.Satellite_num = NEW.Satellite_num
                    )
                    -- If the joint condition is NOT met, RAISE an abort error
                    THEN RAISE (ABORT, 'Integrity violation: Satellite does not exist for the linked snapshot.')
                END;
            END;
        """)

        store_method_results = True
        if store_method_results:
            # --- 6. METHODS TABLE ---
            # Stores the different estimation methods used.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Estimation_Methods (
                    Method_ID       INTEGER PRIMARY KEY AUTOINCREMENT,
                    Method_Name     TEXT    NOT NULL,
                    Parameters_JSON TEXT    -- JSON string of the parameters
                );
            """)
            print("Table 'Estimation_Methods' created.")

            # --- 7. ESTIMATION RESULTS TABLE ---
            # Stores the results of applying estimation methods to MC samples.

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS Estimation_Results (
                    -- Primary Key Columns
                    MC_Sample_ID  INTEGER NOT NULL,
                    Method_ID     INTEGER NOT NULL,

                    -- Required Data (3-element numpy array stored as a BLOB)
                    Error    BLOB    NOT NULL, -- x,y,z in NED about truth error = estimated - truth

                    -- Optional Data (BLOBs and TEXT)
                    Results_data    TEXT    NULL,            -- For JSON string (data and/or metadata for BLOB)
                    Results_blob        BLOB    NULL,        -- For any binary data
                           
                    -- Define the Composite Primary Key
                    PRIMARY KEY (MC_Sample_ID, Method_ID)
                           
                    -- Foreign Keys
                    FOREIGN KEY (MC_Sample_ID)
                        REFERENCES MC_Samples (MC_Sample_ID),
                    FOREIGN KEY (Method_ID)
                        REFERENCES Estimation_Methods (Method_ID)
                );
            """)
            print("Table 'Estimation_Results' created.")

        # Commit all changes and close the connection
        conn.commit()
        print("Database creation complete and changes saved.")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

def insert_real_data(conn: sqlite3.Connection,
                     dataset_name: str,
                     truth_filename: str,
                     input_filename: str) -> bool:
    '''
    Inserts real measurement data into the database from provided files.

    Args:
        conn (sqlite3.Connection): An active database connection object.
        dataset_name (str): Name of the dataset (e.g., 'Chemnitz').
        truth_filename (str): Path to the truth data file (e.g. 'Chemnitz_GT.txt').
        input_filename (str): Path to the measurement input file. (e.g. 'Chemnitz_Input.txt').

    Returns:
        bool: True if data insertion was successful, False otherwise.
    '''
    try:
        cursor = conn.cursor()
        snapshot_time_sync = {}

        with open(truth_filename, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) >= 5 and parts[0] == 'point3':
                    try:
                        time_value = float(parts[1])
                        x_loc = float(parts[2])
                        y_loc = float(parts[3])
                        z_loc = float(parts[4])
                    except:
                        print("Skipping line due to invalid number format: %s", line.strip(), exc_info=True)
                        continue
                    cursor.execute("""
                        INSERT INTO Snapshots (Dataset, Time, True_Loc_x, True_Loc_Y, True_Loc_Z)
                        VALUES (?, ?, ?, ?, ?)
                    """, (dataset_name, time_value, x_loc, y_loc, z_loc))
                    new_snapshot_id = cursor.lastrowid
                    snapshot_time_sync[time_value] = new_snapshot_id
                else:
                    print("Invalid line in GT: %s", line.strip(), exc_info=True)
        conn.commit()
        print('Created snapshots from ground truth data.')

        with open(input_filename, 'r') as file:
            sat_num = 0
            snapshot_ID = None
            curr_time = 0
            line_counter = 0
            mc_sample_id = None

            for line in file:
                line_counter += 1
                if line_counter % 100 == 0:
                    print(f"processing line {line_counter} of measurements", end='\r', flush=True)
                parts = line.split()
                if len(parts) >= 7 and parts[0] == 'pseudorange3':
                    try:
                        time_value = float(parts[1])
                        pseudorange_meas = float(parts[2])
                        sat_x = float(parts[4])
                        sat_y = float(parts[5])
                        sat_z = float(parts[6])
                    except:
                        print("Error parsing pseudorange line")
                        continue

                    # Detect a new snapshot
                    if time_value != curr_time or snapshot_ID is None:
                        if snapshot_ID is not None:
                            if sat_num < 4:
                                print("Warning: Less than 4 satellites found for time", curr_time)
                                conn.rollback()
                                cursor.execute("""
                                    DELETE FROM Snapshots
                                    WHERE Snapshot_ID = ?
                                """, (snapshot_ID,))
                            conn.commit()

                        cursor.execute("BEGIN TRANSACTION")
                        curr_time = time_value
                        sat_num = 0
                        snapshot_ID = snapshot_time_sync.get(curr_time, -1)
                        if snapshot_ID == -1:
                            print("Warning, pseudorange measurement found for time", curr_time, "with no associated ground truth")
                            # Rollback the begun transaction to keep DB consistent
                            conn.rollback()
                            snapshot_ID = None
                            mc_sample_id = None
                            continue
                        else:
                            cursor.execute("""
                                INSERT INTO MC_Samples (Snapshot_ID, MC_Run_ID)
                                VALUES (?, ?)
                            """, (snapshot_ID, 1))
                            mc_sample_id = cursor.lastrowid

                    # Add satellite location
                    cursor.execute("""
                        INSERT INTO Satellite_Locations (Snapshot_ID, Satellite_num, Loc_X, Loc_Y, Loc_Z)
                        VALUES (?, ?, ?, ?, ?)
                    """, (snapshot_ID, sat_num, sat_x, sat_y, sat_z))

                    # Add measurement
                    cursor.execute("""
                        INSERT INTO Measurements (MC_Sample_ID, Satellite_num, Pseudorange)
                        VALUES (?, ?, ?)
                    """, (mc_sample_id, sat_num, pseudorange_meas))

                    sat_num += 1

        # Final check for last snapshot
        if sat_num < 4 and snapshot_ID is not None:
            print("Warning: Less than 4 satellites found for time", curr_time)
            conn.rollback()
            cursor.execute("""
                DELETE FROM Snapshots
                WHERE Snapshot_ID = ?
            """, (snapshot_ID,))
        conn.commit()
        print(f"{dataset_name} data successfully added to database.")
        cursor.execute("SELECT COUNT(*) FROM Snapshots")
        count = cursor.fetchone()[0]
        print(f"Total snapshots in database: {count}")
        return True

    except sqlite3.IntegrityError as e:
        print(f"\nIntegrity Error: Failed to insert data (perhaps key already exists). {e}")
        return False
    except sqlite3.Error as e:
        print(f"\nAn SQL error occurred: {e}")
        return False
    finally:
        cursor.close()


if __name__ == "__main__":
    db_name = "meas_data.db"
    dataset_name = 'Chemnitz'
    conn = sqlite3.connect(db_name)
    # Run the unit test for L2 and pseudorange data 
    sample_ids = get_snapshot_ids(conn, dataset_name=dataset_name)
    

    try:
        data = get_snapshot_data(conn, list([5]))[0] # To pick a random one
    except Exception as e:
        print(f"Nothing retrieved from database: {e}")
    else:
        true_loc, satellites = data
        print('True loc of ',true_loc)
        print('Satellites:\n',satellites)
    