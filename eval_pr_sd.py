import numpy as np
import matplotlib.pyplot as plt
import meas_db_utils as mdu
import comp_utils as cu
import sqlite3

import numpy as np

def mad_sigma_estimate(data):
    """
    Calculates a robust estimate of the standard deviation (sigma)
    using the Median Absolute Deviation (MAD).
    (A Gemini method.  Looks reasonable, so using it.)

    Args:
        data (np.ndarray or list): The array of scalar values.

    Returns:
        float: The robust estimate of the standard deviation.
    """
    # 1. Convert to numpy array for efficient calculation
    data = np.asarray(data)

    # 2. Calculate the Median (M) of the data
    M = np.median(data)

    # 3. Calculate the absolute differences from the median: |x_i - M|
    absolute_differences = np.abs(data - M)

    # 4. Calculate the Median of these absolute differences (the MAD)
    MAD = np.median(absolute_differences)

    # 5. Calculate the robust sigma estimate:
    # The constant 1.4826 (often 1.4826, 1.4828, or just 1.483) 
    # is the scaling factor to make the MAD a consistent estimator for 
    # the standard deviation of a *normally distributed* dataset.
    robust_sigma_estimate = 1.4826 * MAD

    return robust_sigma_estimate

def compute_pseudorange_errors(db_name = 'meas_data.db', out_file_name='pseudorange_errors.npz'):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT DISTINCT Dataset FROM Snapshots")
        datasets = [row[0] for row in cursor.fetchall()]
        print("Available datasets:", datasets)
    except sqlite3.OperationalError as e:
        print("Error querying snapshot table:", e)
    finally:
        cursor.close()

    pr_errors = {}
    for dataset in datasets:
        print('Starting dataset',dataset)
        try:
            sample_ids = mdu.get_mc_sample_ids(conn, 1, dataset)
            truths = mdu.get_mc_sample_truths(conn, sample_ids)
            measurements = mdu.get_mc_samples_measurements(conn, sample_ids)
        except Exception as e:
            print("Error thrown getting data from database",e)
        true_prs = [cu.compute_snapshot_pseudoranges((truth,meas[:,1:])) \
            for truth,meas in zip(truths,measurements)]
        dataset_errors = []
        for snapshot_truth,snapshot_meas in zip(true_prs,measurements):
            snapshot_errors = snapshot_truth-snapshot_meas[:,0]
            dataset_errors.extend(snapshot_errors - np.mean(snapshot_errors)) # Adjust for the receiver clock... :)
        pr_errors[dataset] = dataset_errors
    try:
        np.savez_compressed('pseudorange_errors.npz',
                            **{k: np.asarray(v) for k, v in pr_errors.items()})
        print("Saved pseudorange errors to pseudorange_errors.npz")
    except Exception as e:
        print("Error saving pseudorange errors:", e)
    finally:
        conn.close()

if __name__ == "__main__":
    # Get a whole bunch of snapshots and compute the error in pseudorange for each one
    db_name = "meas_data.db"
    error_storage_file = 'pseudorange_errors.npz'
    first_time = False
    if first_time:
        compute_pseudorange_errors(db_name, error_storage_file)
    ps_errors = np.load(error_storage_file)
    combined_list = []
    sigmas = {}
    for name in ps_errors.files:
        arr = np.asarray(ps_errors[name]).ravel()
        if arr.size == 0:
            continue
        combined_list.extend(arr)
        bins = 100 if arr.size >= 100 else arr.size
        sigma_est = mad_sigma_estimate(arr)
        sigmas[name] = float(sigma_est)
        plt.figure()
        plt.hist(arr, bins=bins, edgecolor='black')
        plt.title(f'Pseudorange errors - {name} ($\sigma$ = {sigma_est:.2f})')
        plt.xlabel('Error (m)')
        plt.ylabel('Count')
        plt.grid(True)
        plt.tight_layout()

    if combined_list:
        arr = np.array(combined_list)
        bins = 100 if len(combined_list) >= 100 else len(combined_list)
        sigma_est = mad_sigma_estimate(arr)
        sigmas['combined'] = float(sigma_est)
        plt.figure()
        plt.hist(arr, bins=bins, edgecolor='black')
        plt.title(f'Pseudorange errors - combined ($\sigma$ = {sigma_est})')
        plt.xlabel('Error (m)')
        plt.ylabel('Count')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("No error arrays found in the file.")
    
    print('Sigma data is:',sigmas)