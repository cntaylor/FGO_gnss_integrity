import sqlite3
import time

def insert_snapshot_data(db_name="simulation_data.db"):
    """
    Demonstrates inserting one snapshot with two Monte Carlo runs, 
    including nested satellite, pseudorange, and result data.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # --- 1. Define Data for ONE Snapshot ---
    current_time = time.time()
    true_loc = (40.7, -74.0, 10.0) # ECEF (X, Y, Z)
    sat_locs = [
        (10000.0, 5000.0, 1000.0), # Sat Index 0
        (11000.0, 6000.0, 1200.0)  # Sat Index 1
    ]
    
    # --- 2. Insert into Snapshots Table ---
    cursor.execute(
        "INSERT INTO Snapshots (Dataset_ID, Time, True_Location_X, True_Location_Y, True_Location_Z) VALUES (?, ?, ?, ?, ?)",
        ("Run_A_Nov23", current_time, true_loc[0], true_loc[1], true_loc[2])
    )
    snapshot_id = cursor.lastrowid # Get the ID of the new snapshot

    # --- 3. Insert into Satellites Table ---
    for i, (x, y, z) in enumerate(sat_locs):
        cursor.execute(
            "INSERT INTO Satellites (Snapshot_ID, Satellite_Index, Loc_X, Loc_Y, Loc_Z) VALUES (?, ?, ?, ?, ?)",
            (snapshot_id, i, x, y, z)
        )

    # --- 4. Define and Insert Monte Carlo Runs ---
    
    # --- MC Run 1 Data ---
    mc_run_1 = {
        'iteration_num': 1,
        'real_sim': 1, # True (real)
        'pseudoranges': [20000.1, 21000.5], # PR for Sat 0, PR for Sat 1
        'results': [
            { # Result 1.1
                'technique': 'WLS',
                'params': '{"dop_limit": 5.0}',
                'fault_declared': 0, # False
                'pl': 15.0,
                'est_loc': (40.71, -74.01, 10.1)
            }
        ]
    }
    
    # --- MC Run 2 Data ---
    mc_run_2 = {
        'iteration_num': 2,
        'real_sim': 0, # False (simulated)
        'pseudoranges': [20001.0, 21002.0],
        'results': [
            { # Result 2.1
                'technique': 'RAIM',
                'params': '{"threshold": 0.5}',
                'fault_declared': 1, # True
                'pl': 12.0,
                'est_loc': (40.68, -73.98, 9.9)
            },
            { # Result 2.2 (multiple results per MC run)
                'technique': 'Kalman',
                'params': '{"Q": 0.1}',
                'fault_declared': 0, # False
                'pl': 8.0,
                'est_loc': (40.70, -74.00, 10.0)
            }
        ]
    }

    # Process MC Run 1 and its children
    for mc_data in [mc_run_1, mc_run_2]:
        
        # Insert into MC_Iterations
        cursor.execute(
            "INSERT INTO MC_Iterations (Snapshot_ID, MC_Iteration_Num, Real_or_Simulated) VALUES (?, ?, ?)",
            (snapshot_id, mc_data['iteration_num'], mc_data['real_sim'])
        )
        mc_id = cursor.lastrowid # Get the ID of the new MC iteration

        # Insert into Pseudoranges
        for i, pr_val in enumerate(mc_data['pseudoranges']):
            cursor.execute(
                "INSERT INTO Pseudoranges (MC_ID, Satellite_Index, Value) VALUES (?, ?, ?)",
                (mc_id, i, pr_val)
            )

        # Insert into Results
        for result in mc_data['results']:
            cursor.execute(
                """
                INSERT INTO Results (MC_ID, Technique, Parameters_Used, Fault_Declared, Protection_Level, Est_Loc_X, Est_Loc_Y, Est_Loc_Z) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (mc_id, result['technique'], result['params'], result['fault_declared'], result['pl'],
                 result['est_loc'][0], result['est_loc'][1], result['est_loc'][2])
            )

    conn.commit()
    conn.close()
    print("\nSample snapshot data successfully inserted.")


# Run the data insertion example
insert_snapshot_data()