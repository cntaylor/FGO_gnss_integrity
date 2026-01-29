import numpy as np
import pickle as pkl
import sqlite3
import meas_db_utils as mdu
import r3f

''' This file is a complement to test_estimation.py.
    It pulls in the results_file and errors_file and
    does further analysis on the results of previous runs.   
'''


def get_errors(results_file, errors_file):
    with open(results_file, 'rb') as f:
        results = pkl.load(f)
    with open(errors_file, 'rb') as f:
        errors = pkl.load(f)
    return results, errors

def analyze_outliers(results, thresholds, true_outliers=None):
    '''
    Take in the outlier selections from ARAIM and the weighting factors
    from the FGO and compare them.  Will return a data structure with
    4 lists.  
        0. When the FGO and ARAIM have the same outliers
        1. When FGO is a subset of ARAIM, 
        2. When ARAIM is a subset of FGO
        3. When neither is a subset of the other.  
    If truth is not None, it will also return lists comparing the truth with the 
    results from FGO and ARAIM outlier determinations (same 4 lists)

    Args:
        results: dictionary of results from the ARAIM and FGO methods
        thresholds: dictionary of thresholds for the FGO method 
            (keyed by method name, same as in results).  Uses .1 if key is not found
        true_outliers: list of truth outliers  per epoch(only obtainable for simulated data)

    Returns:
        A dictionary with keys same as results (without ARAIM or L2), each with the 4 lists above.
        Also, may have "key"_truth which compares methods (including ARAIM) with truth (same 4 lists, but truth rather than ARAIM)
    '''
    araim_outliers = results['ARAIM'][3]
    araim_sets = [set(araim_outliers[i]) for i in range(len(araim_outliers))]
    going_out = {}

    if true_outliers is not None:
        assert len(true_outliers) == len(araim_outliers), "ARAIM and truth need to have the same number of results"

    for key in results.keys():
        if key != 'ARAIM' and key != 'L2' and key != 'truth': 
            # print(key)
            fgo_outliers = results[key][3] # Fourth element in the list
            assert len(fgo_outliers) == len(araim_outliers), "ARAIM and FGO methods need to have the same number of results"
            # Take the output and convert into sets of outliers, using a threshold
            threshold = thresholds.get(key,.1)
            # print("Using threshold of", threshold)
            fgo_sets = [set(np.where(fgo_outliers[i]<threshold)[0].tolist()) for i in range(len(fgo_outliers))]
            # # Used to look more closely at individual cases.
            # print(f'{key} outliers: {fgo_sets[4:8]}')
            # print(f'ARAIM outliers: {araim_sets[4:8]}')
            # Now do the comparison
            going_out[key] = [[],[],[],[]]
            for i, araim, fgo in zip(range(len(araim_outliers)), araim_sets, fgo_sets):
                if araim == fgo:
                    going_out[key][0].append(i)
                elif fgo.issubset(araim):
                    going_out[key][1].append(i)
                elif araim.issubset(fgo):
                    going_out[key][2].append(i)
                else:
                    going_out[key][3].append(i)
            if true_outliers is not None:
                new_key = key + '_truth'
                going_out[new_key] = [[],[],[],[]]
                            # Take the output and convert into sets of outliers, using a threshold
                for i, truth, fgo in zip(range(len(true_outliers)), true_outliers, fgo_sets):   
                    if truth == fgo:
                        going_out[new_key][0].append(i)
                    elif fgo.issubset(truth):
                        going_out[new_key][1].append(i)
                    elif truth.issubset(fgo):
                        going_out[new_key][2].append(i)
                    else:
                        going_out[new_key][3].append(i)
        elif key == 'ARAIM' and true_outliers is not None:
            new_key = key + '_truth'
            going_out[new_key] = [[],[],[],[]]
            for i, truth, araim in zip(range(len(true_outliers)), true_outliers, araim_sets):
                if truth == araim:  
                    going_out[new_key][0].append(i)
                elif araim.issubset(truth):
                    going_out[new_key][1].append(i)
                elif truth.issubset(araim):
                    going_out[new_key][2].append(i)
                else:
                    going_out[new_key][3].append(i)
    return going_out

if __name__ == '__main__':
    ''' Comment pasted from test_estimation...
    '''
    # The parameters requested by the user:
        # run_id: 
        # - 1 = real data
        # - 2 = no outliers
        # - 3 = one outlier
        # - 4 = two outliers
        # - 5 = three outliers
        # - 6 = four outliers
        # - 7->11, Big outliers (10,000 samples)
    run_id = 6
    base_name = 'FourOutliers' # Should be the name of the file and, if real data, the DATASET
    results_file = base_name + '_results.pkl'
    errors_file = base_name + '_errors.pkl'
    results, errors = get_errors(results_file, errors_file)
    simulated_data = True
    conn = sqlite3.connect('meas_data.db')
    if simulated_data:
        # Get the true outliers from the database for comparison
        epochs = mdu.get_mc_sample_ids(conn, run_id)
        true_outliers = mdu.get_mc_sample_outliers(conn, epochs)
        # Turn true_outliers into sets of indices
        true_outlier_sets = [set(np.where(true_outliers[i])[0].tolist()) for i in range(len(true_outliers))]
    else:
        epochs = mdu.get_mc_sample_ids(conn, run_id, dataset_name=base_name)
        true_outlier_sets = None
    true_positions = mdu.get_mc_sample_truths(conn, epochs)
    
    ##### All the data is now read in. Now to process and put out information.
    # Find 3D errors
    errors_3d_enu = {}
    for key in results.keys():
        if key == 'truth':
            continue
        errors_3d_enu[key] = np.zeros_like(results[key][0])
        for i in range(len(results[key][0])):
            errors_3d_enu[key][i] = r3f.ecef_to_tangent(results[key][0][i],true_positions[i], ned=False)
    # And compute the chi-squared value for each error
    # Initialize the dictionary
    chi_sq = {}
    for key in errors_3d_enu.keys():
        if key != "ARAIM":
            chi_sq[key] = np.full(len(errors_3d_enu[key]), np.nan)

    # And keep track of vpl,hpl results.
    vpl_valid = 0
    hpl_valid = 0
    for key in errors_3d_enu.keys():
        if key == "ARAIM": 
            for i in range(len(errors_3d_enu[key])):
                hpl,vpl = results[key][2][i]
                vert_error = abs(errors_3d_enu[key][i][2])
                horz_error = errors_3d_enu[key][i][0]**2 + errors_3d_enu[key][i][1]**2
                vpl_valid += 1 if vert_error <= vpl else 0
                hpl_valid += 1 if horz_error <= hpl else 0

            continue
        for i in range(len(errors_3d_enu[key])):
            chi_sq[key][i] = errors_3d_enu[key][i] @ np.linalg.inv(results[key][2][i][:3,:3]) @ errors_3d_enu[key][i]
    
    # Now let's look at the outliers
    thresholds = {'GemanMcClure':.2, 'Huber':.4, 'Cauchy':.5}
    set_comparisons = analyze_outliers(results, thresholds, true_outlier_sets)
    
    # Now find out the errors & number of runs for each set:
    error_per_set = {}
    runs_per_set = {}
    for key in set_comparisons.keys():
        error_per_set[key] = []
        runs_per_set[key] = []
        # How to access the errors dict
        errors_key = key if "_truth" not in key else key[:-6]
        for i in range(4):
            if len(set_comparisons[key][i]) > 0:
                error_per_set[key].append(np.mean(errors[errors_key][set_comparisons[key][i]]))
                if errors_key != "ARAIM":
                    runs_per_set[key].append(np.mean(results[errors_key][4][set_comparisons[key][i]]))
                else:
                    runs_per_set[key].append(np.nan)
            else:
                error_per_set[key].append(np.nan)
                runs_per_set[key].append(np.nan)

    # To start, let's print out computational information
    print(f"---------For Dataset {base_name}------------")
    print ("Average number of lstsq solves...")
    for key in results.keys():
        if key == "L2" or key == 'truth':
            continue
        avg_solves = np.mean(results[key][4])
        print(f'{key:<18}: {avg_solves}')

    ## Did this test and it really didn't show any difference between what type of sets, so not printing anymore
    # print("\nRuns required broken out by set")
    # print('{:<25}{:<15}{:<15}{:<15}{:<15}'.format('key', 'same', 'subset', 'superset', 'neither'))
    # print('-'*85)
    # sorted_keys = sorted(runs_per_set.keys(), key=lambda x: (x.endswith('_truth'), x))
    # for key in sorted_keys:
    #     print('{:<25}{:<5}{:<10.3}{:<5}{:<10.3}{:<5}{:<10.3}{:<5}{:<10.3}'.format(key, \
    #                                                                               len(set_comparisons[key][0]), runs_per_set[key][0], 
    #                                                                               len(set_comparisons[key][1]), runs_per_set[key][1],
    #                                                                               len(set_comparisons[key][2]), runs_per_set[key][2],
    #                                                                               len(set_comparisons[key][3]), runs_per_set[key][3]))
    print('\n')

    # Now print out the results
    print("Overall error statistics:")
    for key in errors:
        print(f"  {key:<17} :   Average: {np.mean(errors[key]):7.3f} meters,",\
              f" Std Dev: {np.std(errors[key]):7.3f} meters,",
              f" Max: {np.max(errors[key]):7.3f} meters")
    print() # Add a line

    print("Outlier groups and the average error in each group")
    print('{:<25}{:<15}{:<15}{:<15}{:<15}'.format('key', 'same', 'subset', 'superset', 'neither'))
    print('-'*90)
    sorted_keys = sorted(set_comparisons.keys(), key=lambda x: (x.endswith('_truth'), x))
    for key in sorted_keys:
        print('{:<25}{:<6}{:<10.3}{:<6}{:<10.3}{:<6}{:<10.3}{:<6}{:<10.3}'.format(key, \
                                                                                  len(set_comparisons[key][0]), error_per_set[key][0], 
                                                                                  len(set_comparisons[key][1]), error_per_set[key][1],
                                                                                  len(set_comparisons[key][2]), error_per_set[key][2],
                                                                                  len(set_comparisons[key][3]), error_per_set[key][3]))
    print('\n')

    # Now find out the chi-squared errors for each set:
    chisq_per_set = {}
    for key in set_comparisons.keys():
        if 'ARAIM' in key:
            # Can't do ARAIM because it doesn't generate a covariance.  Want to test if it exceed protection level?
            continue
        chisq_per_set[key] = []
        # How to access the errors dict
        chisq_key = key if "_truth" not in key else key[:-6]
        for i in range(4):
            if len(set_comparisons[key][i]) > 0:
                chisq_per_set[key].append(np.mean(chi_sq[chisq_key][set_comparisons[key][i]]))
            else:
                chisq_per_set[key].append(np.nan)
    

    print("ANEES is...")
    for key in chi_sq.keys():
        print(f'{key:<18}: {np.mean(chi_sq[key])}')
    print('\n')

    print("Outlier groups and the average chi squared in each group")
    print('{:<25}{:<15}{:<15}{:<15}{:<15}'.format('key', 'same', 'subset', 'superset', 'neither'))
    print('-'*85)
    sorted_keys = sorted(set_comparisons.keys(), key=lambda x: (x.endswith('_truth'), x))
    for key in sorted_keys:
        if 'ARAIM' in key:
            continue
        print('{:<25}{:<6}{:<10.3}{:<6}{:<10.3}{:<6}{:<10.3}{:<6}{:<10.3}'.format(key, \
                                                                                  len(set_comparisons[key][0]), chisq_per_set[key][0], 
                                                                                  len(set_comparisons[key][1]), chisq_per_set[key][1],
                                                                                  len(set_comparisons[key][2]), chisq_per_set[key][2],
                                                                                  len(set_comparisons[key][3]), chisq_per_set[key][3]))


    # # Let's look at some of the comparative sets between trunc_Gauss and ARAIM.  They should be the same, I think...
    # araim_outliers = results['ARAIM'][2]
    # fgo_outliers = results['gnc_trunc_Gauss'][2]
    # max_cnt = 10
    # print(f"Let's show {max_cnt} that are the same")
    # curr_cnt = 0
    # for same_idx in set_comparisons['gnc_trunc_Gauss'][0]:
    #     print ('------------------- index: ', same_idx,'-------------------')
    #     print(f'gnc_trunc_Gauss: {np.where(fgo_outliers[same_idx] < .5)[0].tolist()}')
    #     print(f'ARAIM: {araim_outliers[same_idx]}')
    #     curr_cnt += 1
    #     if curr_cnt == max_cnt:
    #         break
    # # print('*********************************')
    # print("Let's show 4 that are supersets of ARAIM")
    # curr_cnt=0
    # for super_idx in set_comparisons['gnc_trunc_Gauss'][2]:   
    #     print ('------------------- index:', super_idx,'-------------------')
    #     print(f'gnc_trunc_Gauss: {np.where(fgo_outliers[super_idx] < .5)[0].tolist()}')
    #     print(f'ARAIM: {araim_outliers[super_idx]}')
    #     curr_cnt += 1
    #     if curr_cnt == max_cnt:
    #         break
    # # print(set_comparisons)
    