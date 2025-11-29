import meas_db_utils as mdu
import sqlite3
import numpy as np
import comp_utils as cu
import r3f
from FG_estimation import snapshot_fgo
from ARAIM_estimation import snapshot_ARAIM, init_ARAIM
import matplotlib.pyplot as plt
import pickle

def validate_estimation(conn, run_id, dataset_name, test_params, methods,
                        results_file = "results.pkl",
                        errors_file = "errors.pkl",
                        plot_res = True):
    """
    Test out different techniques compare it against L2 and truth
    """
    print(f"--- Starting Validation for Run {run_id} ({dataset_name}) ---")

    # 1. Get MC_Sample_IDs
    # Assumes MC_run_ID=1 is the 'real data' run as specified.
    sample_ids = mdu.get_mc_sample_ids(conn, run_id, dataset_name=dataset_name)  # Change here for smaller sets for testing
    
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
    results = {method: [] for method in methods}
    results["truth"] = [truth_list] # All other methods have a list of multiple entries.  This makes
                                    # truth look more like the others (easier to plot among other things)
    results["L2"] = []
    results["L2"].append(np.zeros((len(measurements_list),3))) # position estimates
    results["L2"].append(np.zeros(len(measurements_list))) # time offsets
    results["L2"].append([None] * len(measurements_list)) # outlier info
    results["L2"].append([None] * len(measurements_list)) # covariance
    for method in methods:
        if method == "ARAIM":
            # ARAIM doesn't return the timing, so it's a special case
            results["ARAIM"].append(np.zeros((len(measurements_list),3)))
            results["ARAIM"].append(None) #timing
            results["ARAIM"].append([None] * len(measurements_list)) #outlier info
            results["ARAIM"].append([None] * len(measurements_list)) # PLs
        else:
            results[method].append(np.zeros((len(measurements_list),3))) # position estimates
            results[method].append(np.zeros(len(measurements_list))) # timing offsets
            results[method].append([None] * len(measurements_list)) # outlier info
            results[method].append([None] * len(measurements_list)) # covariance
            results[method].append(np.zeros(len(measurements_list),dtype=int)) # num_iterations
            

    for i,measurements_array in enumerate(measurements_list):
        l2_est_loc = cu.estimate_l2_location(measurements_array)
        results["L2"][0][i] = l2_est_loc[0] # position
        results["L2"][1][i] = l2_est_loc[1] # timing offset
        y,J = cu.compute_residual_and_jacobian(measurements_array[:,0], l2_est_loc[0], measurements_array[:,1:], l2_est_loc[1], compute_Jacobian=True)
        results["L2"][3][i] = np.linalg.inv(J.T @ J) * test_params['base_sigma']**2
        for method in methods:
            if method == "ARAIM":
                print("Running method:", method, "for sample", i)
                results[method][0][i], _, \
                    results[method][2][i], results[method][3][i] = \
                    snapshot_ARAIM(measurements_array)
            else:
                print("Running method:", method)
                tmp_params = test_params.copy()
                rcf_method = method
                if "gnc" in method:
                    tmp_params["gnc"] = True
                    rcf_method = rcf_method.removeprefix("gnc_")
                else:
                    tmp_params["gnc"] = False
                if "double" in method: # Double the cut-off values, see the effects.
                    tmp_params["huber_k"] *= 2.0
                    tmp_params["cauchy_c"] *= 2.0
                    tmp_params["geman_c"] *= 2.0
                    tmp_params["trunc_k"] *= 2.0
                    rcf_method = rcf_method.removesuffix("_double")
                tmp_params["rcf"] = rcf_method
                results[method][0][i], results[method][1][i], \
                    results[method][2][i], results[method][3][i], \
                    results[method][4][i] = \
                    snapshot_fgo(measurements_array,params=tmp_params)
        
        
    # Convert lists to NumPy arrays for easy calculation
    truth_array = np.array(truth_list)

    # 4. Compute Errors
    
    # Calculate difference in X, Y, Z axes
    result_errors = dict.fromkeys(methods)
    result_errors["L2"] = np.array([x for x in results["L2"][0]]).reshape(-1,3) - truth_array.reshape(-1,3)
    for method in methods:
        result_errors[method] = np.array([x for x in results[method][0]]).reshape(-1,3) - truth_array

    # Calculate Total Error (Euclidean Distance)
    total_errors = {}
    for key in result_errors:
        total_errors[key] = np.linalg.norm(result_errors[key],axis=1) # (N x 1) array of total errors
    
    # 5. Print out the statistics
    print("\n--- STATISTICAL RESULTS ---")
    print(f"Total Error (Magnitude) Statistics:")
    for key in result_errors:
        print(f"  {key:>17} :   Average: {np.mean(total_errors[key]):7.3f} meters,",\
              f" Std Dev: {np.std(total_errors[key]):7.3f} meters,",
              f" Max: {np.max(total_errors[key]):7.3f} meters")

    # 6.  Save all the results to a file
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    with open(errors_file, 'wb') as f:
        pickle.dump(total_errors, f)

    # 7. Plotting (Matplotlib is required)
    # plot errors
    if plot_res:
        for key in result_errors:
            plt.plot(total_errors[key], label=key)
        plt.title('Total Error Magnitude (3D)')
        plt.xlabel('Sample Index')
        plt.grid(True)
        plt.legend()

        # plot LLA results
        plt.figure()
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        for key in results:
            curr_locs = r3f.ecef_to_geodetic(results[key][0])
            ax1.plot(curr_locs[:, 1], curr_locs[:, 0], label=key)
            ax2.plot(curr_locs[:, 2], label=key)

        ax1.set_xlabel('Longitude (rad)')
        ax1.set_ylabel('Latitude (rad)')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Height (meters)')

        fig1.suptitle(f'Latitude/Longitude: Truth vs estimated (Run {run_id}, {dataset_name})')
        fig2.suptitle(f'Height: Truth vs estimated (Run {run_id}, {dataset_name})')

        ax1.legend()
        ax2.legend()

        ax1.grid(True)
        ax2.grid(True)

        ax1.axis('equal')
        plt.show()

if __name__ == '__main__':
    import meas_db_utils as mdu
    import sqlite3


    DB_FILE = "meas_data.db"
    try:
        conn = sqlite3.connect(DB_FILE)

        # The parameters requested by the user:
        # run_id: 
        # - 1 = real data
        # - 2 = no outliers, no noise (for debugging)
        # - 3 = no outliers
        # - 4 = one outlier
        # - 5 = two outliers
        # - 6 = three outliers
        # - 7 = four outliers
        MC_RUN_ID = 1   # Assuming 1 is your real data or target run
        DATASET = 'UrbanNav_Harsh'  # Example dataset name
        filename_base = DATASET #'OneOutlier'
        test_params = {
            "rcf" : "Cauchy",  # Robust cost function
            "base_sigma" : 5.0,  # Base measurement noise standard deviation (meters)
            "gnc" : False,    # Graduated non-convexity
            "max_iters" : 50, # How many Gauss-Newton steps to allow
            "tolerance" : 1e-4,
            "geman_c": 3.0,    # Tuning constant for Geman-McClure
            "huber_k": 1.345,  # Tuning constant for Huber
            "cauchy_c": 2.385, # Tuning constant for Cauchy
            "trunc_k": 3.0,
            # more araim parameters
            "araim_set_covariance" : True,
            "araim_set_fault_prob" : True,
            "fault_prob" : 0.001,
            "araim_use_bias_for_PL" : False,
            "max_bias" : 0.0

        }

        validate_estimation(conn, MC_RUN_ID, DATASET, test_params, \
                                ["ARAIM","Huber","Cauchy","GemanMcClure","gnc_trunc_Gauss","gnc_GemanMcClure"],\
                                results_file = filename_base+"_results.pkl",\
                                errors_file = filename_base+"_errors.pkl", \
                                plot_res = False)
#
    except sqlite3.Error as e:
        print(f"A fatal database error occurred: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()