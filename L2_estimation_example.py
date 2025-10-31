import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from meas_db_utils import get_MC_sample_ids, get_MC_samples_meas, get_MC_samples_truth, L2_est_location
import r3f

def validate_estimation(cursor, run_id, dataset_name):
    """
    Performs the full validation loop: data retrieval, estimation, error calculation,
    and prepares results for plotting and statistics.
    """
    print(f"--- Starting Validation for Run {run_id} ({dataset_name}) ---")

    # 1. Get MC_Sample_IDs
    # Assumes MC_run_ID=1 is the 'real data' run as specified.
    sample_id_tuples = get_MC_sample_ids(cursor, run_id, dataset_name=dataset_name)
    
    if not sample_id_tuples:
        print("No MC samples found for the specified run and dataset.")
        return

    # Flatten the list of tuples into a list of IDs
    sample_ids = [s[0] for s in sample_id_tuples]
    print(f"Found {len(sample_ids)} unique MC samples.")

    to_process = sample_ids # Can cut it down for debugging here if wanted
    # 2. Get Measurements and Truth Data
    try:
        # get_measurements returns a list of Nx4 NumPy arrays
        # each entry in the numpy array has [pseudorange, sat_X, sat_Y, sat_Z]
        measurements_list = get_MC_samples_meas(cursor, to_process) 
        
        # get_MC_samples_truth returns a list of 3-element NumPy arrays
        truth_list = get_MC_samples_truth(cursor, to_process)
        
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
    estimated_locations = np.zeros((len(measurements_list),3))
    time_offsets = np.zeros(len(measurements_list))
    
    for i,measurements_array in enumerate(measurements_list):
        est_loc = L2_est_location(measurements_array)
        estimated_locations[i] = est_loc[0].copy()
        time_offsets[i] = est_loc[1]
        
    # Convert lists to NumPy arrays for easy calculation
    truth_array = np.array(truth_list)

    # 4. Compute Errors
    
    # Calculate difference in X, Y, Z axes
    errors_xyz = estimated_locations - truth_array  # (N x 3) array of errors
    
    # Calculate Total Error (Magnitude or Euclidean Distance)
    # sqrt(E_x^2 + E_y^2 + E_z^2)
    total_error = np.linalg.norm(errors_xyz, axis=1) # (N x 1) array of total errors

    # 5. Compute Statistics
    
    avg_error = np.mean(total_error)
    std_dev_error = np.std(total_error)
    max_error = np.max(total_error)
    
    # Errors on each axis for individual statistics
    avg_error_xyz = np.mean(errors_xyz, axis=0)
    std_dev_error_xyz = np.std(errors_xyz, axis=0)
    max_error_xyz = np.max(np.abs(errors_xyz), axis=0) # Max absolute error

    print("\n--- STATISTICAL RESULTS ---")
    print(f"Total Error (Magnitude) Statistics:")
    print(f"  Average Error:   {avg_error:.3f} meters")
    print(f"  Std Dev Error:   {std_dev_error:.3f} meters")
    print(f"  Maximum Error:   {max_error:.3f} meters")
    print("-" * 30)
    print(f"Axis-by-Axis Error Statistics (Mean, Std, Max_Abs):")
    print(f"| Axis | Mean Error | Std Dev | Max Abs Error |")
    print(f"|:----:|:----------:|:-------:|:-------------:|")
    print(f"|  X   | {avg_error_xyz[0]:.3f} | {std_dev_error_xyz[0]:.3f} | {max_error_xyz[0]:.3f} |")
    print(f"|  Y   | {avg_error_xyz[1]:.3f} | {std_dev_error_xyz[1]:.3f} | {max_error_xyz[1]:.3f} |")
    print(f"|  Z   | {avg_error_xyz[2]:.3f} | {std_dev_error_xyz[2]:.3f} | {max_error_xyz[2]:.3f} |")
    print("-" * 30)

    # 6. Plotting (Matplotlib is required)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Estimation Error Analysis (Run {run_id}, Dataset: {dataset_name})', fontsize=16)

    # Plot 1: X-Axis Error
    axes[0, 0].plot(errors_xyz[:, 0], label='X-Error', color='r')
    axes[0, 0].set_title('Error in X-Axis')
    axes[0, 0].set_ylabel('Error (meters)')
    axes[0, 0].grid(True)

    # Plot 2: Y-Axis Error
    axes[0, 1].plot(errors_xyz[:, 1], label='Y-Error', color='g')
    axes[0, 1].set_title('Error in Y-Axis')
    axes[0, 1].grid(True)

    # Plot 3: Z-Axis Error
    axes[1, 0].plot(errors_xyz[:, 2], label='Z-Error', color='b')
    axes[1, 0].set_title('Error in Z-Axis')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Error (meters)')
    axes[1, 0].grid(True)

    # Plot 4: Total Error Magnitude
    axes[1, 1].plot(total_error, label='Total Error', color='k')
    axes[1, 1].set_title('Total Error Magnitude (3D)')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.figure()
    plt.plot(time_offsets)
    plt.title('Time Offsets')
    plt.ylabel('Time (seconds*c = meters)')

    plt.figure(figsize=(8, 6))

    try:
        # Try common r3f function names
        est_lla = np.array([r3f.ecef_to_geodetic(pt) for pt in estimated_locations])
        truth_lla = np.array([r3f.ecef_to_geodetic(pt) for pt in truth_array])
    except Exception as e:
        raise RuntimeError("r3f conversion failed") from e

    # Plot Lat/Lon: longitude on x-axis, latitude on y-axis

    plt.plot(truth_lla[:, 1], truth_lla[:, 0], 'k.-', label='Truth')
    plt.plot(est_lla[:, 1], est_lla[:, 0], 'r.-', label='Estimated')
    plt.xlabel('Longitude (rad)')
    plt.ylabel('Latitude (rad')
    plt.title(f'Latitude/Longitude: Truth vs Estimated (Run {run_id}, {dataset_name})')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plt.figure()
    plt.plot(truth_lla[:, 2], 'k.-', label='Truth')
    plt.plot(est_lla[:, 2], 'r.-', label='Estimated')  
    plt.xlabel('Sample Index')
    plt.ylabel('Height (meters)')
    plt.show()

    return avg_error, std_dev_error, max_error

# --- MAIN EXECUTION BLOCK (Conceptual) ---

if __name__ == '__main__':
    DB_FILE = "meas_data.db"
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # The parameters requested by the user:
        MC_RUN_ID = 1      # Assuming 1 is your real data or target run
        DATASET = 'Berlin_Potsdamer'  # Example dataset name

        validate_estimation(cursor, MC_RUN_ID, DATASET)

    except sqlite3.Error as e:
        print(f"A fatal database error occurred: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()