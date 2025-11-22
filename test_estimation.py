import meas_db_utils as mdu
import sqlite3
import numpy as np
import comp_utils as cu
import r3f
import time
from FG_estimation import snapshot_fgo
from ARAIM_estimation import snapshot_ARAIM, init_ARAIM
import matplotlib.pyplot as plt

def validate_estimation(conn, run_id, dataset_name, test_params):
    """
    Test out FGO with params and compare it against L2 and truth
    """
    print(f"--- Starting Validation for Run {run_id} ({dataset_name}) ---")

    # 1. Get MC_Sample_IDs
    # Assumes MC_run_ID=1 is the 'real data' run as specified.
    sample_ids = mdu.get_mc_sample_ids(conn, run_id, dataset_name=dataset_name)  # Limit to first 100 for speed
    
    if not sample_ids:
        print("No MC samples found for the specified run and dataset.")
        return

    to_process = sample_ids # Can cut it down for debugging here if wanted
    # 2. Get Measurements and Truth Data
    try:
        # get_measurements returns a list of Nx4 NumPy arrays
        # each entry in the numpy array has [pseudorange, sat_X, sat_Y, sat_Z]
        measurements_list = mdu.get_mc_samples_measurements(conn, to_process) 
        
        # get_MC_samples_truth returns a list of 3-element NumPy arrays
        truth_list = mdu.get_mc_sample_truths(conn, to_process)
        
        # Sanity Check: Ensure the lists are the same length
        if len(measurements_list) != len(truth_list):
            print("ERROR: Mismatch in length between measurements and truth data.")
            return

    except ValueError as e:
        print(f"Data retrieval failed due to missing IDs: {e}")
        return
    except Exception as e:
        print(f"Data retrieval failed: {e}")
        return

    # 3. Run Estimation and Store Results
    init_ARAIM(test_params)
    l2_estimated_locations = np.zeros((len(measurements_list),3))
    l2_time_offsets = np.zeros(len(measurements_list))
    fgo_estimated_locations = np.zeros_like(l2_estimated_locations)
    fgo_time_offsets = np.zeros_like(l2_time_offsets)
    fgo_outlier_info = [None] * len(measurements_list) # For speed
    fgo_covariance = [None] * len(measurements_list) # For speed
    
    araim_est_locs = np.zeros_like(l2_estimated_locations)
    araim_outlier_info = [None] * len(measurements_list) # For speed
    araim_PLs = [None] * len(measurements_list) # For speed
    for i,measurements_array in enumerate(measurements_list):
        l2_est_loc = cu.estimate_l2_location(measurements_array)
        l2_estimated_locations[i] = l2_est_loc[0]
        l2_time_offsets[i] = l2_est_loc[1]
        fgo_estimated_locations[i], fgo_time_offsets[i],\
           fgo_outlier_info[i], fgo_covariance[i] = \
           snapshot_fgo(measurements_array,params=test_params)
        araim_est_locs[i], _, \
            araim_outlier_info[i], araim_PLs[i] = snapshot_ARAIM(measurements_array)
        
        
    # Convert lists to NumPy arrays for easy calculation
    truth_array = np.array(truth_list)

    # 4. Compute Errors
    
    # Calculate difference in X, Y, Z axes
    l2_errors_xyz = l2_estimated_locations - truth_array  # (N x 3) array of errors
    fgo_errors_xyz = fgo_estimated_locations - truth_array  # (N x 3) array of errors
    araim_errors_xyz = araim_est_locs - truth_array  # (N x 3) array of errors

    # Calculate Total Error (Magnitude or Euclidean Distance)
    # sqrt(E_x^2 + E_y^2 + E_z^2)
    l2_total_error = np.linalg.norm(l2_errors_xyz, axis=1) # (N x 1) array of total errors
    fgo_total_error = np.linalg.norm(fgo_errors_xyz, axis=1) # (N x 1) array of total errors
    araim_total_error = np.linalg.norm(araim_errors_xyz, axis=1) # (N x 1) array of total errors

    # 5. Compute Statistics

    l2_avg_error = np.mean(l2_total_error)
    l2_std_dev_error = np.std(l2_total_error)
    l2_max_error = np.max(l2_total_error)
    fgo_avg_error = np.mean(fgo_total_error)
    fgo_std_dev_error = np.std(fgo_total_error)
    fgo_max_error = np.max(fgo_total_error)
    araim_avg_error = np.mean(araim_total_error)
    araim_std_dev_error = np.std(araim_total_error)
    araim_max_error = np.max(araim_total_error)

    # Errors on each axis for individual statistics
    l2_avg_error_xyz = np.mean(l2_errors_xyz, axis=0)
    l2_std_dev_error_xyz = np.std(l2_errors_xyz, axis=0)
    l2_max_error_xyz = np.max(np.abs(l2_errors_xyz), axis=0) # Max absolute error
    fgo_avg_error_xyz = np.mean(fgo_errors_xyz, axis=0)
    fgo_std_dev_error_xyz = np.std(fgo_errors_xyz, axis=0)
    fgo_max_error_xyz = np.max(np.abs(fgo_errors_xyz), axis=0) # Max absolute error
    araim_avg_error_xyz = np.mean(araim_errors_xyz, axis=0)
    araim_std_dev_error_xyz = np.std(araim_errors_xyz, axis=0)
    araim_max_error_xyz = np.max(np.abs(araim_errors_xyz), axis=0) # Max absolute error

    # Print out the statistics
    print("\n--- STATISTICAL RESULTS ---")
    print(f"Total Error (Magnitude) Statistics:")
    print(f"  Average Error:   L2: {l2_avg_error:7.3f} meters, FGO: {fgo_avg_error:7.3f} meters, ARAIM: {araim_avg_error:7.3f} meters")
    print(f"  Std Dev Error:   L2: {l2_std_dev_error:7.3f} meters, FGO: {fgo_std_dev_error:7.3f} meters, ARAIM: {araim_std_dev_error:7.3f} meters")
    print(f"  Maximum Error:   L2: {l2_max_error:7.3f} meters, FGO: {fgo_max_error:7.3f} meters, ARAIM: {araim_max_error:7.3f} meters")
    print("-" * 30)
    print(f"Axis-by-Axis Error Statistics L2 (Mean, Std, Max_Abs):")
    print(f"| Axis | Mean Error |  Std Dev  | Max Abs Error |")
    print(f"|:----:|:----------:|:---------:|:-------------:|")
    print(f"|  X   | {l2_avg_error_xyz[0]:10.3f} | {l2_std_dev_error_xyz[0]:9.3f} | {l2_max_error_xyz[0]:13.3f} |")
    print(f"|  Y   | {l2_avg_error_xyz[1]:10.3f} | {l2_std_dev_error_xyz[1]:9.3f} | {l2_max_error_xyz[1]:13.3f} |")
    print(f"|  Z   | {l2_avg_error_xyz[2]:10.3f} | {l2_std_dev_error_xyz[2]:9.3f} | {l2_max_error_xyz[2]:13.3f} |")
    print("-" * 50)
    print(f"Axis-by-Axis Error Statistics FGO (Mean, Std, Max_Abs):")
    print(f"| Axis | Mean Error |  Std Dev  | Max Abs Error |")
    print(f"|:----:|:----------:|:---------:|:-------------:|")
    print(f"|  X   | {fgo_avg_error_xyz[0]:10.3f} | {fgo_std_dev_error_xyz[0]:9.3f} | {fgo_max_error_xyz[0]:13.3f} |")
    print(f"|  Y   | {fgo_avg_error_xyz[1]:10.3f} | {fgo_std_dev_error_xyz[1]:9.3f} | {fgo_max_error_xyz[1]:13.3f} |")
    print(f"|  Z   | {fgo_avg_error_xyz[2]:10.3f} | {fgo_std_dev_error_xyz[2]:9.3f} | {fgo_max_error_xyz[2]:13.3f} |")
    print("-" * 50)
    print(f"Axis-by-Axis Error Statistics ARAIM (Mean, Std, Max_Abs):")
    print(f"| Axis | Mean Error |  Std Dev  | Max Abs Error |")
    print(f"|:----:|:----------:|:---------:|:-------------:|")
    print(f"|  X   | {araim_avg_error_xyz[0]:10.3f} | {araim_std_dev_error_xyz[0]:9.3f} | {araim_max_error_xyz[0]:13.3f} |")
    print(f"|  Y   | {araim_avg_error_xyz[1]:10.3f} | {araim_std_dev_error_xyz[1]:9.3f} | {araim_max_error_xyz[1]:13.3f} |")
    print(f"|  Z   | {araim_avg_error_xyz[2]:10.3f} | {araim_std_dev_error_xyz[2]:9.3f} | {araim_max_error_xyz[2]:13.3f} |")

    # 6. Plotting (Matplotlib is required)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Estimation Error Analysis (Run {run_id}, Dataset: {dataset_name})', fontsize=16)
    # Plot 1: X-Axis Error
    axes[0, 0].plot(l2_errors_xyz[:, 0], label='L2', color='r')
    axes[0, 0].plot(fgo_errors_xyz[:, 0], label='FGO', color='b', linestyle='--')
    axes[0, 0].plot(araim_errors_xyz[:, 0], label='araim', color='g', linestyle='--')
    axes[0, 0].set_title('Error in X-Axis')
    axes[0, 0].set_ylabel('Error (meters)')
    axes[0, 0].grid(True)

    # Plot 2: Y-Axis Error
    axes[0, 1].plot(l2_errors_xyz[:, 1], label='L2', color='r')
    axes[0, 1].plot(fgo_errors_xyz[:, 1], label='FGO', color='b', linestyle='--')
    axes[0, 1].plot(araim_errors_xyz[:, 1], label='araim', color='g', linestyle='--')
    axes[0, 1].set_title('Error in Y-Axis')
    axes[0, 1].grid(True)

    # Plot 3: Z-Axis Error
    axes[1, 0].plot(l2_errors_xyz[:, 2], label='L2', color='r')
    axes[1, 0].plot(fgo_errors_xyz[:, 2], label='FGO', color='b', linestyle='--')
    axes[1, 0].plot(araim_errors_xyz[:, 2], label='araim', color='g', linestyle='--')
    axes[1, 0].set_title('Error in Z-Axis')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Error (meters)')
    axes[1, 0].grid(True)

    # Plot 4: Total Error Magnitude
    axes[1, 1].plot(l2_total_error, label='L2', color='r')
    axes[1, 1].plot(fgo_total_error, label='FGO', color='b', linestyle='--')
    axes[1, 1].plot(araim_total_error, label='araim', color='g', linestyle='--')
    axes[1, 1].set_title('Total Error Magnitude (3D)')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].grid(True)
    fig.legend()

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    
    plt.figure()
    plt.plot(l2_time_offsets, 'r.-', label='L2 Time Offsets')
    plt.plot(fgo_time_offsets, 'b--', label='FGO Time Offsets')
    # plt.plot(araim_time_offsets, 'g--', label='ARAIM Time Offsets')
    plt.title('Time Offsets')
    plt.ylabel('Time (seconds*c = meters)')
    plt.legend()

    plt.figure(figsize=(8, 6))

    try:
        # Convert locaitons to lat/lon
        l2_est_lla = np.array([r3f.ecef_to_geodetic(pt) for pt in l2_estimated_locations])
        fgo_est_lla = np.array([r3f.ecef_to_geodetic(pt) for pt in fgo_estimated_locations])
        araim_est_lla = np.array([r3f.ecef_to_geodetic(pt) for pt in araim_est_locs])    
        truth_lla = np.array([r3f.ecef_to_geodetic(pt) for pt in truth_array])
    except Exception as e:
        raise RuntimeError("r3f conversion failed") from e

    # Plot Lat/Lon: longitude on x-axis, latitude on y-axis

    plt.plot(truth_lla[:, 1], truth_lla[:, 0], 'k.-', label='Truth')
    plt.plot(l2_est_lla[:, 1], l2_est_lla[:, 0], 'r.-', label='L2')
    plt.plot(fgo_est_lla[:, 1], fgo_est_lla[:, 0], 'b.-', label='FGO')
    plt.plot(araim_est_lla[:, 1], araim_est_lla[:, 0], 'g.-', label='araim')
    plt.xlabel('Longitude (rad)')
    plt.ylabel('Latitude (rad)')
    plt.title(f'Latitude/Longitude: Truth vs Estimated (Run {run_id}, {dataset_name})')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plt.figure()
    plt.plot(truth_lla[:, 2], 'k.-', label='Truth')
    plt.plot(l2_est_lla[:, 2], 'r.-', label='L2')
    plt.plot(fgo_est_lla[:, 2], 'b.-', label='FGO')
    plt.plot(araim_est_lla[:, 2], 'g.-', label='araim')
    plt.xlabel('Sample Index')
    plt.ylabel('Height (meters)')
    plt.title(f'Height: Truth vs Estimated (Run {run_id}, {dataset_name})')
    plt.show()

# --- As a test, run a dataset and compare the RCF results with the basic L2 optimization ---

if __name__ == '__main__':
    import meas_db_utils as mdu
    import sqlite3


    DB_FILE = "meas_data.db"
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # The parameters requested by the user:
        MC_RUN_ID = 1      # Assuming 1 is your real data or target run
        DATASET = 'Berlin_Potsdamer'  # Example dataset name
        test_params = {
            "rcf" : "Cauchy",  # Robust cost function
            "base_sigma" : 14.4,  # Base measurement noise standard deviation (meters)
            "gnc" : False,    # Graduated non-convexity
            "max_iters" : 50, # How many Gauss-Newton steps to allow
            "tolerance" : 1e-4,
            "geman_c": 2.0,    # Tuning constant for Geman-McClure
            # more araim parameters
            "araim_set_covariance" : True,
            "araim_set_fault_prob" : True,
            "fault_prob" : 0.01,
            "araim_use_bias_for_PL" : False,
            "max_bias" : 0.0

        }

        validate_estimation(conn, MC_RUN_ID, DATASET, test_params)

    except sqlite3.Error as e:
        print(f"A fatal database error occurred: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()