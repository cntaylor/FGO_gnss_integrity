import sqlite3
import setup_db
import numpy as np

# This adds data from
# https://github.com/TUC-ProAut/libRSF/tree/master/datasets/
# Need the _GT.txt and _Input.txt files


def add_measurement_data(db_name="meas_data.db",
                            create_new_db=True,
                            dataset_name='Chemnitz',
                            truth_filename='Chemnitz_GT.txt',
                            input_filename='Chemnitz_Input.txt'):
    """
    Add measurements from truth_filename and input_filename into the database db_name.
    If create_new_db is True, create the database schema first using setup_db.create_measurement_database.
    """
    if create_new_db:
        # In case it hasn't been done already
        setup_db.create_measurement_database(db_name)

    conn = None
    try:
        conn = sqlite3.connect(db_name)
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
                        print(f"Skipping line due to invalid number format: {line.strip()}")
                        continue
                    cursor.execute("""
                        INSERT INTO Snapshots (Dataset, Time, True_Loc_x, True_Loc_Y, True_Loc_Z)
                        VALUES (?, ?, ?, ?, ?)
                    """, (dataset_name, time_value, x_loc, y_loc, z_loc))
                    new_snapshot_id = cursor.lastrowid
                    snapshot_time_sync[time_value] = new_snapshot_id
                else:
                    print("invalid line in GT")
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
        if conn:
            conn.close()


if __name__ == "__main__":
    # Chemnitz dataset
    add_measurement_data(
        db_name="meas_data.db",
        create_new_db=True,
        dataset_name='Chemnitz',
        truth_filename='Chemnitz_GT.txt',
        input_filename='Chemnitz_Input.txt'
    )
    # UrbanNav Data set.  It has "Deep", "Harsh", "Medium", "Obaida", and "Shinjuku" datasets
    #Deep
    add_measurement_data(
        db_name="meas_data.db",
        create_new_db=False,
        dataset_name='UrbanNav_Deep',
        truth_filename='UrbanNav_HK_Deep_Urban_GT.txt',
        input_filename='UrbanNav_HK_Deep_Urban_Input.txt'
    )
    #Harsh
    add_measurement_data(
        db_name="meas_data.db",
        create_new_db=False,
        dataset_name='UrbanNav_Harsh',
        truth_filename='UrbanNav_HK_Harsh_Urban_GT.txt',
        input_filename='UrbanNav_HK_Harsh_Urban_Input.txt'
    )
    #Medium
    add_measurement_data(
        db_name="meas_data.db",
        create_new_db=False,
        dataset_name='UrbanNav_Medium',
        truth_filename='UrbanNav_HK_Medium_Urban_GT.txt',
        input_filename='UrbanNav_HK_Medium_Urban_Input.txt'
    )
    #Obaida
    add_measurement_data(
        db_name="meas_data.db",
        create_new_db=False,
        dataset_name='UrbanNav_Obaida',
        truth_filename='UrbanNav_TK_Obaida_GT.txt',
        input_filename='UrbanNav_TK_Obaida_Input.txt'
    )
    #Shinjuku
    add_measurement_data(
        db_name="meas_data.db",
        create_new_db=False,
        dataset_name='UrbanNav_Shinjuku',
        truth_filename='UrbanNav_TK_Shinjuku_GT.txt',
        input_filename='UrbanNav_TK_Shinjuku_Input.txt'
    )

    # And the SmartLoc dataset:  Frankfurt_Westend, Frankfurt_Main, Berlin_Gendarmenmarkt, and Berlin_Potsdamer
    #Frankfurt_Westend
    add_measurement_data(
        db_name="meas_data.db",
        create_new_db=False,
        dataset_name='Frankfurt_Westend',
        truth_filename='Frankfurt_Westend_Tower_GT.txt',
        input_filename='Frankfurt_Westend_Tower_Input.txt'
    )
    #Frankfurt_Main
    add_measurement_data(
        db_name="meas_data.db",
        create_new_db=False,
        dataset_name='Frankfurt_Main',
        truth_filename='Frankfurt_Main_Tower_GT.txt',
        input_filename='Frankfurt_Main_Tower_Input.txt'
    )
    #Berlin_Gendarmenmarkt
    add_measurement_data(
        db_name="meas_data.db",
        create_new_db=False,
        dataset_name='Berlin_Gendarmenmarkt',
        truth_filename='Berlin_Gendarmenmarkt_GT.txt',
        input_filename='Berlin_Gendarmenmarkt_Input.txt'
    )
    #Berlin_Potsdamer
    add_measurement_data(
        db_name="meas_data.db",
        create_new_db=False,
        dataset_name='Berlin_Potsdamer',
        truth_filename='Berlin_Potsdamer_Platz_GT.txt',
        input_filename='Berlin_Potsdamer_Platz_Input.txt'
    )