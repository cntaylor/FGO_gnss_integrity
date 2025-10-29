import meas_db_utils as mdu
import sqlite3 
import numpy as np
import matplotlib.pyplot as plt
import json

if __name__ == "__main__":
    # Connect to the database
    db_name = "chemnitz_data.db"
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    samples=10000
    try:
        # Do we want to create a new MC run or use an old one?
        new_mc_run_id = False
        old_mc_run_id = 2   
        if new_mc_run_id:
            mc_description = "MC simulation: Gaussian noise + outliers"
            noise_std_dev = 5.0  # meters
            outlier_fraction = 0.10  # 10% outliers
            outlier_sd = 100.0  # meters
            mc_params = {
                "simulated": True,
                "noise_std_dev": noise_std_dev,
                "outlier_fraction": outlier_fraction,
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
            if noise_std_dev == -1 or outlier_fraction == -1 or outlier_sd == -1:
                raise ValueError(f"Invalid Monte Carlo parameters in MC_run {mc_run_id}.")
        # Generate Monte Carlo samples
        possible_snapshot_ids = mdu.get_snapshot_ids(cursor)
        # Have a long list of snapshots.  Now to randomly select from them
        selected_snapshot_ids = np.random.choice(
            possible_snapshot_ids, size=samples, replace=True
        )
        # sqlite doesn't like np.int64 types, so convert to native Python int
        selected_snapshot_ids = [int(sid) for sid in selected_snapshot_ids]
        # Now to get the data needed to generate measurements
        noiseless_data = mdu.noiseless_model(cursor, selected_snapshot_ids)
        to_database_list = 
        # Take each sample and add noise/outliers to generate measurements
        for snapshot_id,noiseless_pseudoranges in zip(selected_snapshot_ids,noiseless_data):
            outliers = np.random.rand(len(noiseless_pseudoranges)) < outlier_fraction
            noise = np.random.normal(0, noise_std_dev, size=len(noiseless_pseudoranges))
            noise[outliers] = np.random.normal(0, outlier_sd, size=np.sum(outliers))
            noisy_pseudoranges = noiseless_pseudoranges + noise
            # Store the generated measurements in the database
            to_database_list.append((snapshot_id, noisy_pseudoranges, outliers))
        
        mdu.add_MC_samples(conn, mc_run_id, to_database_list)
    except Exception as e:
        print(f"Error occurred: {e}")
        conn.rollback()
    # Close the database connection
    conn.close()