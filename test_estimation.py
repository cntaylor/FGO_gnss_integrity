import meas_db_utils as mdu
import sqlite3
import numpy as np
import comp_utils as cu
import r3f
from FG_estimation import single_epoch_fgo
from ARAIM_estimation import epoch_ARAIM, init_ARAIM
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool
from tqdm import tqdm

def single_epoch_results(args):
    i, measurements_array, methods, test_params = args
    local_results = {} # will have index + all the methods (and "L2") in the results
    local_results["index"] = i
    l2_est_loc = cu.estimate_l2_location(measurements_array)
    # compute the covariance of the L2 estimation
    y,J = cu.compute_residual_and_jacobian(measurements_array[:,0], l2_est_loc[0], measurements_array[:,1:], l2_est_loc[1], compute_Jacobian=True)
    l2_covariance  = np.linalg.inv(J.T @ J) * test_params['base_sigma']**2
    # local_results["L2"] = (position, timing offset, covariance)
    local_results["L2"] = (l2_est_loc[0], l2_est_loc[1], l2_covariance)
    
    for method in methods:
        if method == "ARAIM":
            local_results[method] = epoch_ARAIM(measurements_array)
        else:
            # Change the parameters depending on the method name
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
            if "onePointFive" in method: # scale the cut-off values, see the effects.
                tmp_params["huber_k"] *= 1.5
                tmp_params["cauchy_c"] *= 1.5
                tmp_params["geman_c"] *= 1.5
                tmp_params["trunc_k"] *= 1.5
                rcf_method = rcf_method.removesuffix("_onePointFive")
            tmp_params["rcf"] = rcf_method
            # actually run the method
            local_results[method] = \
                single_epoch_fgo(measurements_array,params=tmp_params)
    return local_results
    


def validate_estimation(conn, run_id, dataset_name, test_params, methods,
                        results_file = "results.pkl",
                        errors_file = "errors.pkl",
                        plot_res = True, 
                        run_parallel = True):
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
        measurements_list = mdu.get_mc_sample_measurements(conn, to_process) 
        
        # get_mc_sample_truths returns a list of 3-element NumPy arrays
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
    # Create data structures to store results  (init lists so it doesn't take so long to run)
    results = {method: [] for method in methods}
    results["truth"] = [truth_list] # All other methods have a list of multiple entries.  This makes
                                    # truth look more like the others (easier to plot among other things)
    results["L2"] = []
    results["L2"].append(np.zeros((len(measurements_list),3))) # position estimates
    results["L2"].append(np.zeros(len(measurements_list))) # time offsets
    results["L2"].append([None] * len(measurements_list)) # covariance
    results["L2"].append([None] * len(measurements_list)) # outlier info
    for method in methods:
        if method == "ARAIM":
            # ARAIM doesn't return the timing, so it's a special case
            results["ARAIM"].append(np.zeros((len(measurements_list),3)))
            results["ARAIM"].append(None) #timing
            results["ARAIM"].append([None] * len(measurements_list)) # PLs
            results["ARAIM"].append([None] * len(measurements_list)) #outlier info
            results["ARAIM"].append(np.zeros(len(measurements_list), dtype=int)) # num_iterations
        else:
            results[method].append(np.zeros((len(measurements_list),3))) # position estimates
            results[method].append(np.zeros(len(measurements_list))) # timing offsets
            results[method].append([None] * len(measurements_list)) # covariance
            results[method].append([None] * len(measurements_list)) # outlier info
            results[method].append(np.zeros(len(measurements_list),dtype=int)) # num_iterations
            
    # Actually run the results
    if run_parallel:
        tasks = [(i, measurements_array, methods, test_params) for i,measurements_array in enumerate(measurements_list)]
        with Pool(initializer=init_ARAIM, initargs=(test_params,)) as pool:
            result_iterator = pool.imap(single_epoch_results, tasks, chunksize=10)
            for single_result in tqdm(result_iterator, total=len(tasks), desc="Running epochs"):
                i = single_result["index"]
                for j in range(3):
                    results["L2"][j][i] = single_result["L2"][j]
                for method in methods:
                    for j in range(len(results[method])):
                        if method != "ARAIM" or j != 1:
                            results[method][j][i] = single_result[method][j]
    else:
        print('Running methods:')
        print(methods)
        for i,measurements_array in enumerate(measurements_list):
            print(f"Running epoch {i} ...................")
            single_results = single_epoch_results((i, measurements_array,methods, test_params))
            for j in range(3):
                results["L2"][j][i] = single_results["L2"][j]
            for method in methods:
                for j in range(len(results[method])):
                    if method != "ARAIM" or j != 1:
                        results[method][j][i] = single_results[method][j]
        
        
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
        print(f"  {key:>25} :   Average: {np.mean(total_errors[key]):7.3f} meters,",\
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
        # - 2 = no outliers
        # - 3 = one outlier
        # - 4 = two outliers
        # - 5 = three outliers
        # - 6 = four outliers
        # - 7-11 == big (10k) of 0-4 outliers
        test_params = {
            "rcf" : "Cauchy",  # Robust cost function
            "base_sigma" : 14.4,  # Base measurement noise standard deviation (meters)
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
        methods_compare = ["ARAIM",
            "Huber", 
            "Cauchy", 
            "GemanMcClure",
            "gnc_GemanMcClure",
            "gnc_trunc_Gauss"
            ]
        run_parallel = True # set to False for debugging
        sim = True # Different data is recorded depending on sim or not
        if sim:
            sim_run_ids  = [ 2, 3, 4, 5, 6]
            sim_filenames = ["NoOutliers", "OneOutlier", "TwoOutliers", "ThreeOutliers", "FourOutliers"]
            # sim_run_ids  = [ 7, 8, 9, 10, 11]
            # sim_filenames = ["Big_NoOutliers", "Big_OneOutlier", "Big_TwoOutliers", "Big_ThreeOutliers", "Big_FourOutliers"]
            # sim_run_ids = [2,3,4,5,6,7,8,9,10,11]
            # sim_filenames = ["NoOutliers", "OneOutlier", "TwoOutliers", "ThreeOutliers", "FourOutliers",
            #                  "Big_NoOutliers", "Big_OneOutlier", "Big_TwoOutliers", "Big_ThreeOutliers", "Big_FourOutliers"]
            # sim_run_ids = [2]
            # sim_filenames = ["NoOutliers"]

            test_params["base_sigma"] = 5.0 # For simulated data...  Comment out for real data!
            for run_id, filename in zip(sim_run_ids, sim_filenames):
                validate_estimation(conn, run_id, None, test_params, \
                                    methods_compare,\
                                    results_file = filename+"_results.pkl",\
                                    errors_file = filename+"_errors.pkl", \
                                    plot_res = False, run_parallel=run_parallel)
        else:
            run_id = 1
            test_params["base_sigma"] = 14.4
            dataset_names = ['UrbanNav_Medium'] #mdu.get_dataset_names(conn) # -- runs all the datasets at once
            pass_list = []
            fail_list = []
            for dataset in dataset_names:
                print("Running for dataset:", dataset)
                try:
                    validate_estimation(conn, run_id, dataset, test_params, methods_compare,
                                    results_file=dataset+"_results.pkl",
                                    errors_file=dataset+"_errors.pkl",
                                    plot_res=False, run_parallel=run_parallel)
                    pass_list.append(dataset)
                except:
                    print("Failed for dataset:", dataset)
                    fail_list.append(dataset)

            print("Passed:", pass_list)
            print("Failed:", fail_list)
    except sqlite3.Error as e:
        print(f"A fatal database error occurred: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()