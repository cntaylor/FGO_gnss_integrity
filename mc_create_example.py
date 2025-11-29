import meas_db_utils as mdu
import comp_utils as cu
import sqlite3 
import numpy as np
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    # Connect to the database
    db_name = "meas_data.db"
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    samples=1000
    try:
        # Do we want to create a new MC run or use an old one?
        new_mc_run_id = True
        old_mc_run_id = 3  
        if new_mc_run_id:
            mc_description = "MC simulation: four outliers"
            noise_std_dev = 5.  # meters
            outlier_set_num = True
            outlier_fraction = 0.0  # 10% outliers 
            outlier_sd = 100.0  # meters
            num_outliers = 4
            mc_params = {
                "simulated": True,
                "noise_std_dev": noise_std_dev,
                "set_num_outliers": outlier_set_num,
                "num_outliers": num_outliers,
                "outlier_fraction": outlier_fraction, # if set_num_outliers is False...
                "outlier_sd": outlier_sd
            }
            # Create a new Monte Carlo run entry
            cursor.execute("""
                INSERT INTO MC_Runs (Description, Parameters_JSON)
                VALUES (?, ?)
            """, (mc_description, json.dumps(mc_params)))  # Store parameters as JSON
            mc_run_id = cursor.lastrowid
            conn.commit()
            print(f"Created MC Run ID: {mc_run_id}")
        else:
            mc_run_id = old_mc_run_id
            print(f"Using existing MC Run ID: {mc_run_id}")
            cursor.execute("""
                SELECT Parameters_JSON FROM MC_Runs WHERE MC_Run_ID = ?
            """, (mc_run_id,))
            mc_params = json.loads(cursor.fetchone()[0])
            noise_std_dev = mc_params.get("noise_std_dev", -1)
            outlier_fraction = mc_params.get("outlier_fraction", -1)
            outlier_sd = mc_params.get("outlier_sd", -1)
            outlier_set_num = mc_params.get("set_num_outliers", False)
            num_outliers = mc_params.get("num_outliers", -1)
            if noise_std_dev == -1 or outlier_sd == -1 or (outlier_set_num and num_outliers == -1) or \
                (not outlier_set_num and outlier_fraction == -1): 
                raise ValueError(f"Invalid Monte Carlo parameters in MC_run {mc_run_id}.")
        # Generate Monte Carlo samples
        possible_snapshot_ids = mdu.get_snapshot_ids(conn)
        # Have a long list of snapshots.  Now to randomly select from them
        selected_snapshot_ids = np.random.choice(
            possible_snapshot_ids, size=samples, replace=True
        )
        # sqlite doesn't like np.int64 types, so convert to native Python int
        selected_snapshot_ids = [int(sid) for sid in selected_snapshot_ids]
        # Now to get the data needed to generate measurements
        snapshot_data = mdu.get_snapshot_data(conn, selected_snapshot_ids)
        # Compute noiseless pseudoranges for each sample
        noiseless_data = cu.compute_list_snapshot_pseudoranges(snapshot_data)
        to_database_list = [None] * len(selected_snapshot_ids)
        # Take each sample and add noise/outliers to generate measurements
        count = 0
        for i in range(len(selected_snapshot_ids)):
            snapshot_id = selected_snapshot_ids[i]
            noiseless_pseudoranges = noiseless_data[i]
            if outlier_set_num:
                choose_outliers = np.random.choice(len(noiseless_pseudoranges), size=num_outliers, replace=False)
                outliers = np.zeros(len(noiseless_pseudoranges), dtype=bool)
                outliers[choose_outliers] = True
            else:
                outliers = np.random.rand(len(noiseless_pseudoranges)) < outlier_fraction
            noise = np.random.normal(0, noise_std_dev, size=len(noiseless_pseudoranges))
            noise[outliers] = np.random.normal(0, outlier_sd, size=np.sum(outliers))
            noisy_pseudoranges = noiseless_pseudoranges + noise
            # Store the generated measurements in the database
            to_database_list[i] = ({'Snapshot_ID': snapshot_id,
                                    'pseudoranges': noisy_pseudoranges,
                                    'is_outlier': outliers})
        print(f"Generated {len(to_database_list)} Monte Carlo samples.")
        print("Inserting samples into database...")
        mdu.insert_mc_samples(conn, mc_run_id, to_database_list)
    except Exception as e:
        print(f"Error occurred: {e}")
        conn.rollback()
    # Close the database connection
    conn.close()