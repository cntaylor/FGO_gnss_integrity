import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import meas_db_utils as mdu
import comp_utils as cu
import r3f

# This software must use FGO to evaluate performance of the following parameters (from the submitted abstract):
#  1.  Different robust cost functions (e.g., Huber, Cauchy, etc., including a truncated Gaussian)
#  2.  Effects of graduated non-convexity (initialization?)

# And the process should generate the following outputs:
#  1.  Position estimates (so errors can be computed and analyzed)
#  2.  Covariance estimates (to see how well they match actual errors)
#  3.  Weighting used for each satellite (to allow further processing to decide how well faults were detected)
#

default_fgo_params = {
    "rcf" : "Huber",  # Robust cost function
    "base_sigma" : 14.4,  # Base measurement noise standard deviation (meters)
    "gnc" : True,    # Graduated non-convexity
    "max_iters" : 50, # How many Gauss-Newton steps to allow
    "tolerance" : 1e-4
}

def huber_weights(residuals, k):
    """
    Huber weights for IRLS.
    residuals : numpy array of residuals (same units as k)
    k : threshold parameter (positive scalar)
    returns: weights array same shape as residuals
    """
    abs_r = np.abs(residuals)
    w = np.ones_like(residuals, dtype=float)
    mask = abs_r > k
    w[mask] = k / abs_r[mask]
    return w

def cauchy_weights(residuals, c):
    """
    Cauchy weights: w = 1 / (1 + (r/c)^2)
    c : tuning constant (positive scalar)
    """
    return 1.0 / (1.0 + (residuals / c) ** 2)

def geman_mcclure_weights(residuals, c):
    """
    Geman-McClure weights (robust, downweights large residuals strongly).
    A common practical form for the IRLS weight is: w = 1 / (1 + (r/c)^2)^2
    c : tuning constant (positive scalar)
    """
    denom = 1.0 + (residuals / c) ** 2
    return 1.0 / (denom ** 2)

def gnc_geman_mcclure_weights(residuals, c, theta):
    """
    GNC Geman-McClure weights.
    residuals : numpy array of residuals
    c : tuning constant (positive scalar)
    theta : GNC parameter (positive scalar)
    returns: weights array
    """
    denom = 1.0 + (residuals / c) ** 2 / theta
    return 1.0 / (denom ** 2)

def trunc_gauss_weights(residuals, cutoff):
    """
    Truncated Gaussian weights.
    - residuals: numpy array of residuals
    - sigma: Gaussian sigma (same units as residuals)
    - cutoff: absolute threshold (same units) after which weight = 0
    Behavior: w = 1 for |r| <= cutoff, otherwise 0.
    """
    abs_r = np.abs(residuals)
    w = np.ones_like(residuals, dtype=float)
    w[abs_r > cutoff] = 0.0
    return w

def get_rcf_weights(residuals, params):
    """
    Dispatch to the appropriate RCF weight function based on params["rcf"].
    residuals : numpy array of residuals
    params : dictionary with keys:
        - "rcf": one of "Huber", "Cauchy", "GemanMcClure", "trunc_Gauss"
        - "base_sigma": base measurement sigma (used to scale tuners)
        optional tuners:
        - "huber_k", "cauchy_c", "geman_c"
        - "trunc_sigma", "trunc_k" (multiplier for cutoff in sigma units)
    returns: weights array
    """
    rcf = params.get("rcf", "Huber")
    base_sigma = float(params.get("base_sigma", 1.0))

    # sensible defaults (multipliers chosen from common practice)
    if rcf == "Huber":
        k = params.get("huber_k", 1.345) * base_sigma
        return huber_weights(residuals, k)
    elif rcf == "Cauchy":
        c = params.get("cauchy_c", 2.385) * base_sigma
        return cauchy_weights(residuals, c)
    elif rcf in ("GemanMcClure", "Geman-McClure", "Geman_McClure"):
        c = params.get("geman_c", 2.0) * base_sigma
        return geman_mcclure_weights(residuals, c)
    elif rcf in ("trunc_Gauss", "TruncGauss", "TruncatedGaussian"):
        # trunc_k is a multiplier applied to base_sigma to produce the cutoff threshold
        trunc_k = float(params.get("trunc_k", 3.0))  # default cutoff at 3*sigma
        cutoff = trunc_k * base_sigma
        return trunc_gauss_weights(residuals, cutoff)
    else:
        raise ValueError(f"Unknown RCF '{rcf}'. Supported: Huber, Cauchy, GemanMcClure, trunc_Gauss")
    
def get_gnc_rcf_weights(residuals, params):
    """
    Dispatch to the appropriate RCF weight function based on params["rcf"].
    residuals : numpy array of residuals
    params : dictionary with keys:
        - "rcf": one of "GemanMcClure"
        - "base_sigma": base measurement sigma (used to scale tuners)
        - "geman_c"
    returns: weights array
    """
    rcf = params.get("rcf", "Huber")
    base_sigma = float(params.get("base_sigma", 1.0))

    # sensible defaults (multipliers chosen from common practice)
    if rcf in ("GemanMcClure", "Geman-McClure", "Geman_McClure"):
        c = params.get("geman_c", 2.0) * base_sigma
        return geman_mcclure_weights(residuals, c)
    elif rcf in ("trunc_Gauss", "TruncGauss", "TruncatedGaussian"):
        # trunc_k is a multiplier applied to base_sigma to produce the cutoff threshold
        trunc_k = float(params.get("trunc_k", 3.0))  # default cutoff at 3*sigma
        cutoff = trunc_k * base_sigma
        return trunc_gauss_weights(residuals, cutoff)
    else:
        raise ValueError(f"Unknown RCF '{rcf}'. Supported: Huber, Cauchy, GemanMcClure, trunc_Gauss")
    
def snapshot_fgo(measurements, params = default_fgo_params):
    """
    Take in an array of measurements (Nx4 numpy array with columns: pseudorange, sat_X, sat_Y, sat_Z)
    and perform the FGO snapshot estimation.  

    Returns: tuple containing
        est_location: Estimated location (X, Y, Z) as a numpy array
        est_time_offset: The time offset used to find the estimated location
        weights: weights used on all pseudo-range (could be used to declare outliers)
        cov: Covariance matrix of the final result (Obtained by weighting of pseudoranges and assuming base_sd on measurements)
    """

    est_location, time_offset = cu.estimate_l2_location(measurements)
    if params["gnc"]: # Not currently working!
        print("GNC not implemented yet...")
        pass
        # Use the Li-Ta approach of setting the original "weight" high enough that all residuals are in the 
        # convex region of the robust cost function, then gradually reduce it.
        # Find the maximum residual from the initial estimate
        if params["rcf"] not in ("GemanMcClure", "Geman-McClure", "Geman_McClure"):
            raise ValueError("GNC is only implemented for Geman-McClure RCF in this example.")
        y = np.zeros(len(measurements))
        for i,meas in enumerate(measurements):
            est_pseudorange = mdu.noiseless_pseudorange(est_location, meas[1:4])
            y[i] = meas[0] - (est_pseudorange + time_offset)
        max_residual = np.max(np.abs(y))/params['base_sigma']
        # Equation 23 from Li-Ta paper
        theta = 3 * max_residual**2 / params.get("geman_c", 2.0)**2
        # To perform GNC, will have two loops.  Outer loop reduces theta, inner loop does IRLS
        while theta > 1.:
            # Inner loop: Perform IRLS
            weights = get_rcf_weights(y, params)
            # Update residuals
            for i,meas in enumerate(measurements):
                est_pseudorange = mdu.noiseless_pseudorange(est_location, meas[1:4])
                y[i] = meas[0] - (est_pseudorange + time_offset)
            max_residual = np.max(np.abs(y))/params['base_sigma']
            theta = 3 * max_residual**2 / params.get("geman_c", 2.0)**2
    else: # not GNC!
        delta_mag = 1000.
        curr_time_offset = time_offset
        num_iters=0
        while delta_mag > params.get('tolerance',.01) and num_iters < params.get('max_iters',50):
            y,J = cu.compute_residual_and_jacobian(measured_pseudoranges=measurements[:,0],
                                                   estimated_location=est_location,
                                                   satellite_locations=measurements[:,1:],
                                                   time_offset=curr_time_offset,
                                                   compute_Jacobian=True)
            weights = get_rcf_weights(residuals = y,
                                      params = params)
            yp = weights * y
            Jp = weights[:, np.newaxis] * J # Element-wise multiply each row by weight
            lstsq_worked=True
            try:
                results = np.linalg.lstsq(Jp,yp)
            except:
                lstsq_worked=False

            if lstsq_worked and results[2] == 4:
                delta = results[0]
                est_location += delta[:3]
                curr_time_offset += delta[3]
                delta_mag = np.linalg.norm(delta)
            else:
                print("Problem with robuts FGO ... No longer of sufficient rank!  Quitting prematurely")
                delta_mag = 0.
            num_iters += 1
    return est_location, time_offset, weights, np.linalg.inv(Jp.T@Jp)

# def run_fgo_estimation(db_name, run_id):
#     """
#     Runs the estimation process for a specific Monte Carlo run.
#     """
#     # Connect to the database
#     conn = sqlite3.connect(db_name)
#     cursor = conn.cursor()

#     try:
#         # First, find out which MC_Sample_IDs correspond to the given run_id
#         sample_ids = get_MC_sample_ids(cursor, run_id)
#     except Exception as e:
#         print(f"Error occurred while retrieving MC_Sample_IDs: {e}")
#         return
    
#     print(f"Retrieved {len(sample_ids)} MC samples to process in run_fgo_estimation.")

#     # Retrieve all the measurements and truth data for these samples at once to minimize DB calls
#     try:
#         measurements = get_MC_samples_meas(cursor, sample_ids)
#         truths = get_MC_samples_truth(cursor, sample_ids)
#     except Exception as e:
#         print(f"Error occurred while retrieving measurements or truths: {e}")
#         return

#     # Main loop just runs through all of the samples found
#     for i in range(len(sample_ids)):
#         mc_sample_id = sample_ids[i]
#         meas_array = measurements[i]
#         truth = truths[i]

#         # Run the L2 estimation
#         est_location, time_offset = snapshot_fgo(meas_array)

#         # Here you would typically store the results back into the database
#         # For this example, we'll just print them
#         print(f"MC_Sample_ID: {mc_sample_id}, Estimated Location: {est_location}, Time Offset: {time_offset}")
#     # Close the database connection
#     conn.close()

def validate_estimation(conn, run_id, dataset_name, fgo_params):
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
    l2_estimated_locations = np.zeros((len(measurements_list),3))
    l2_time_offsets = np.zeros(len(measurements_list))
    fgo_estimated_locations = np.zeros_like(l2_estimated_locations)
    fgo_time_offsets = np.zeros_like(l2_time_offsets)
    fgo_outlier_info = [None] * len(measurements_list) # For speed
    fgo_covariance = [None] * len(measurements_list) # For speed
    for i,measurements_array in enumerate(measurements_list):
        l2_est_loc = cu.estimate_l2_location(measurements_array)
        l2_estimated_locations[i] = l2_est_loc[0]
        l2_time_offsets[i] = l2_est_loc[1]
        fgo_estimated_locations[i], fgo_time_offsets[i],\
           fgo_outlier_info[i], fgo_covariance[i] = \
           snapshot_fgo(measurements_array,params=fgo_params)
        
    # Convert lists to NumPy arrays for easy calculation
    truth_array = np.array(truth_list)

    # 4. Compute Errors
    
    # Calculate difference in X, Y, Z axes
    l2_errors_xyz = l2_estimated_locations - truth_array  # (N x 3) array of errors
    fgo_errors_xyz = fgo_estimated_locations - truth_array  # (N x 3) array of errors
    
    # Calculate Total Error (Magnitude or Euclidean Distance)
    # sqrt(E_x^2 + E_y^2 + E_z^2)
    l2_total_error = np.linalg.norm(l2_errors_xyz, axis=1) # (N x 1) array of total errors
    fgo_total_error = np.linalg.norm(fgo_errors_xyz, axis=1) # (N x 1) array of total errors

    # 5. Compute Statistics

    l2_avg_error = np.mean(l2_total_error)
    l2_std_dev_error = np.std(l2_total_error)
    l2_max_error = np.max(l2_total_error)
    fgo_avg_error = np.mean(fgo_total_error)
    fgo_std_dev_error = np.std(fgo_total_error)
    fgo_max_error = np.max(fgo_total_error)

    # Errors on each axis for individual statistics
    l2_avg_error_xyz = np.mean(l2_errors_xyz, axis=0)
    l2_std_dev_error_xyz = np.std(l2_errors_xyz, axis=0)
    l2_max_error_xyz = np.max(np.abs(l2_errors_xyz), axis=0) # Max absolute error
    fgo_avg_error_xyz = np.mean(fgo_errors_xyz, axis=0)
    fgo_std_dev_error_xyz = np.std(fgo_errors_xyz, axis=0)
    fgo_max_error_xyz = np.max(np.abs(fgo_errors_xyz), axis=0) # Max absolute error

    # Print out the statistics
    print("\n--- STATISTICAL RESULTS ---")
    print(f"Total Error (Magnitude) Statistics:")
    print(f"  Average Error:   L2: {l2_avg_error:7.3f} meters, FGO: {fgo_avg_error:7.3f} meters")
    print(f"  Std Dev Error:   L2: {l2_std_dev_error:7.3f} meters, FGO: {fgo_std_dev_error:7.3f} meters")
    print(f"  Maximum Error:   L2: {l2_max_error:7.3f} meters, FGO: {fgo_max_error:7.3f} meters")
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

    # 6. Plotting (Matplotlib is required)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Estimation Error Analysis (Run {run_id}, Dataset: {dataset_name})', fontsize=16)
    # Plot 1: X-Axis Error
    axes[0, 0].plot(l2_errors_xyz[:, 0], label='L2', color='r')
    axes[0, 0].plot(fgo_errors_xyz[:, 0], label='FGO', color='b', linestyle='--')
    axes[0, 0].set_title('Error in X-Axis')
    axes[0, 0].set_ylabel('Error (meters)')
    axes[0, 0].grid(True)

    # Plot 2: Y-Axis Error
    axes[0, 1].plot(l2_errors_xyz[:, 1], label='L2', color='r')
    axes[0, 1].plot(fgo_errors_xyz[:, 1], label='FGO', color='b', linestyle='--')
    axes[0, 1].set_title('Error in Y-Axis')
    axes[0, 1].grid(True)

    # Plot 3: Z-Axis Error
    axes[1, 0].plot(l2_errors_xyz[:, 2], label='L2', color='r')
    axes[1, 0].plot(fgo_errors_xyz[:, 2], label='FGO', color='b', linestyle='--')
    axes[1, 0].set_title('Error in Z-Axis')
    axes[1, 0].set_xlabel('Sample Index')
    axes[1, 0].set_ylabel('Error (meters)')
    axes[1, 0].grid(True)

    # Plot 4: Total Error Magnitude
    axes[1, 1].plot(l2_total_error, label='L2', color='r')
    axes[1, 1].plot(fgo_total_error, label='FGO', color='b', linestyle='--')
    axes[1, 1].set_title('Total Error Magnitude (3D)')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].grid(True)
    fig.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.figure()
    plt.plot(l2_time_offsets, 'r.-', label='L2 Time Offsets')
    plt.plot(fgo_time_offsets, 'b--', label='FGO Time Offsets')
    plt.title('Time Offsets')
    plt.ylabel('Time (seconds*c = meters)')
    plt.legend()

    plt.figure(figsize=(8, 6))

    try:
        # Try common r3f function names
        l2_est_lla = np.array([r3f.ecef_to_geodetic(pt) for pt in l2_estimated_locations])
        fgo_est_lla = np.array([r3f.ecef_to_geodetic(pt) for pt in fgo_estimated_locations])
        truth_lla = np.array([r3f.ecef_to_geodetic(pt) for pt in truth_array])
    except Exception as e:
        raise RuntimeError("r3f conversion failed") from e

    # Plot Lat/Lon: longitude on x-axis, latitude on y-axis

    plt.plot(truth_lla[:, 1], truth_lla[:, 0], 'k.-', label='Truth')
    plt.plot(l2_est_lla[:, 1], l2_est_lla[:, 0], 'r.-', label='L2')
    plt.plot(fgo_est_lla[:, 1], fgo_est_lla[:, 0], 'b.-', label='FGO')
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
    plt.xlabel('Sample Index')
    plt.ylabel('Height (meters)')
    plt.title(f'Height: Truth vs Estimated (Run {run_id}, {dataset_name})')
    plt.show()

# --- As a test, run a dataset and compare the RCF results with the basic L2 optimization ---

if __name__ == '__main__':
    DB_FILE = "meas_data.db"
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # The parameters requested by the user:
        MC_RUN_ID = 1      # Assuming 1 is your real data or target run
        DATASET = 'Berlin_Potsdamer'  # Example dataset name
        test_fgo_params = {
            "rcf" : "Cauchy",  # Robust cost function
            "base_sigma" : 14.4,  # Base measurement noise standard deviation (meters)
            "gnc" : False,    # Graduated non-convexity
            "max_iters" : 50, # How many Gauss-Newton steps to allow
            "tolerance" : 1e-4,
            "geman_c": 2.0    # Tuning constant for Geman-McClure
        }

        validate_estimation(conn, MC_RUN_ID, DATASET, test_fgo_params)

    except sqlite3.Error as e:
        print(f"A fatal database error occurred: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()