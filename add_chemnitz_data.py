import sqlite3
import setup_db
import numpy as np

# This adds data from
# https://github.com/TUC-ProAut/libRSF/tree/master/datasets/Chemnitz%20City
# Need the Chemnitz_GT.txt and Chemnitz_Input.txt files

# Are you creating a new database or adding to a new one?
create_new_db = True
if create_new_db:
    db_name = "chemnitz_data.db"
    # In case it hasn't been done already
    setup_db.create_measurement_database(db_name)
else:
    db_name = "GNSS_meas.db"

conn = None
try:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    dataset_name = 'Chemnitz'
    gt_filename = 'Chemnitz_GT.txt'
    input_filename = 'Chemnitz_Input.txt'
    snapshot_time_sync = {}
    with open(gt_filename,'r') as file:
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
                    INSERT INTO Snapshots (Dataset, Time, True_Loc_x, True_Loc_Y, True_Loc_Z)
                    VALUES (?, ?, ?, ?, ?)
                """, (dataset_name, time_value, x_loc, y_loc, z_loc))
                # Keep track of the maaping from time_value to snapshot_ID (used for inserting satellites
                new_snapshot_id = cursor.lastrowid
                snapshot_time_sync[time_value] = new_snapshot_id
            else:
                print("invalid line in GT")
    conn.commit()
    with open(input_filename,'r') as file:
        sat_num = 0
        snapshot_ID = None # So that it triggers the "new MC sample" flag below
        sample_ID = None
        curr_time= 0 
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
                # Detect a new snapshot
                if time_value != curr_time or snapshot_ID is None:
                    if snapshot_ID is not None:
                        # Do a test to make sure at least 4 satellites were found for the last snapshot
                        if sat_num < 4:
                            print("Warning: Less than 4 satellites found for time",curr_time)
                            # This should get rid of the MC_sample entry and the associated measurements
                            conn.rollback()
                            # Now to delete the snapshot entry too
                            cursor.execute("""
                                DELETE FROM Snapshots
                                WHERE Snapshot_ID = ?
                            """, (snapshot_ID,))
                        conn.commit()
                    cursor.execute("BEGIN TRANSACTION")
                    curr_time = time_value
                    sat_num = 0
                    snapshot_ID = snapshot_time_sync.get(curr_time,-1)
                    if snapshot_ID == -1:
                        print("Warning, pseudorange measurement found for time",curr_time,"with no associated ground truth")
                        continue
                    else:
                        # Add this "MC sample" into the tracking table
                        cursor.execute("""
                            INSERT INTO MC_Samples (Snapshot_ID, MC_Run_ID)
                            VALUES (?, ?)
                        """,(snapshot_ID, 1)) # 1 = real data
                        mc_sample_id = cursor.lastrowid # Used for putting in measurements
                        # print("Inserted an MC run, ID:",mc_sample_id)
                # Add in specific satellite information
                # Put the satellite location into the database
                cursor.execute("""
                    INSERT INTO Satellite_Locations (Snapshot_ID, Satellite_num, Loc_X, Loc_Y, Loc_Z)
                    VALUES (?, ?, ?, ?, ?)
                """,(snapshot_ID, sat_num, sat_x, sat_y, sat_z))
                # Put the actual pseduorange measurement in
                cursor.execute("""
                    INSERT INTO Measurements (MC_Sample_ID, Satellite_num, Pseudorange)
                    Values (?, ?, ?)
                """, (mc_sample_id, sat_num, pseudorange_meas))
                # Prepare for the next measurements
                sat_num += 1
    # Do a test to make sure at least 4 satellites were found for the last snapshot
    if sat_num < 4:
        print("Warning: Less than 4 satellites found for time",curr_time)
        # This should get rid of the MC_sample entry and the associated measurements
        conn.rollback()
        # Now to delete the snapshot entry too
        cursor.execute("""
            DELETE FROM Snapshots
            WHERE Snapshot_ID = ?
        """, (snapshot_ID,))
    conn.commit()
    print(f"{dataset_name} data successfully added to database.")
    cursor.execute("""
        SELECT COUNT(*) FROM Snapshots
    """)
    count = cursor.fetchone()[0]
    print(f"Total snapshots in database: {count}")

except sqlite3.IntegrityError as e:
    print(f"\nIntegrity Error: Failed to insert data (perhaps key already exists). {e}")
except sqlite3.Error as e:
    print(f"\nAn SQL error occurred: {e}")
finally:
    if conn:
        conn.close()

