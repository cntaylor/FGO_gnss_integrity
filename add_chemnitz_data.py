import sqlite3
import setup_db
import numpy as np

# This adds data from
# https://github.com/TUC-ProAut/libRSF/tree/master/datasets/Chemnitz%20City

db_name = "chemnitz_data.db"
# In case it hasn't been done already
setup_db.create_measurement_database(db_name)


conn = None
try:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    snapshot_num = 0
    dataset_id = 'Chemnitz'
    time_num_sync = []
    with open('Chemnitz_GT.txt','r') as file:
        for line in file:
            # Should split the line up
            parts = line.split()
            if len(parts) >= 5 and parts[0] == 'point3':
                try:
                    time_value = float(parts[1])
                    x_loc = float(parts[2])
                    y_loc = float(parts[3])
                    z_loc = float(parts[4])
                except:
                    print(f"Skipping line due to invalid number format: {line.strip()}")
                    continue
                # Stick truth in to SQL Timesteps table"
                cursor.execute("""
                    INSERT INTO Timesteps (Dataset_ID, Snapshot_num, Time, True_Loc_x, True_Loc_Y, True_Loc_Z)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (dataset_id, snapshot_num, time_value, x_loc, y_loc, z_loc))
                # the index # of each time should be == snapshot_num.  Used for reading real data later.
                time_num_sync.append(time_value)
                snapshot_num += 1
            else:
                print("invalid line in GT")

    with open('Chemnitz_Input.txt','r') as file:
        sat_num = 0
        curr_time=-1. # So that it triggers the "new MC run" flag below
        snapshot_num=-1 
        for line in file:
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
                if time_value != curr_time:
                    curr_time = time_value
                    sat_num = 0
                    snapshot_num += 1
                    if time_num_sync[snapshot_num] != curr_time:
                        snapshot_num = np.searchsorted(time_num_sync, curr_time)
                    # Add this "MC run" into the tracking table
                    cursor.execute("""
                        INSERT INTO MC_Runs (Dataset_ID, Snapshot_num, MC_Param_ID)
                        VALUES (?, ?, ?)
                    """,(dataset_id, snapshot_num, 1)) # 1 = real data
                    mc_run_id = cursor.lastrowid # Used for putting in measurements
                    print("Inserted an MC run, ID:",mc_run_id)
                    if mc_run_id is None:
                        print('mc_run_id returned nothing.  Problem inserting data?')
                # Put the satellite location into the database
                cursor.execute("""
                    INSERT INTO Satellite_Locations (Dataset_ID, Snapshot_num, Satellite_num, Loc_X, Loc_Y, Loc_Z)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,(dataset_id, snapshot_num, sat_num, sat_x, sat_y, sat_z))
                # Put the actual pseduorange measurement in
                cursor.execute("""
                    INSERT INTO Measurements (MC_Run_ID, Satellite_num, Pseudorange)
                    Values (?, ?, ?)
                """, (mc_run_id, sat_num, pseudorange_meas))
                # Prepare for the next measurements
                sat_num += 1
except sqlite3.IntegrityError as e:
    print(f"\nIntegrity Error: Failed to insert data (perhaps key already exists). {e}")
except sqlite3.Error as e:
    print(f"\nAn SQL error occurred: {e}")
finally:
    if conn:
        conn.close()

