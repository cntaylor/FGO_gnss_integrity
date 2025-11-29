import numpy as np
import pickle as pkl
import sqlite3
import meas_db_utils as mdu

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
        true_outliers: list of truth outliers  per snapshot(only obtainable for simulated data)

    Returns:
        A dictionary with keys same as results (without ARAIM or L2), each with the 4 lists above.
        Also, may have "key"_truth which compares methods (including ARAIM) with truth (same 4 lists, but truth rather than ARAIM)
    '''
    araim_outliers = results['ARAIM'][2]
    araim_sets = [set(araim_outliers[i]) for i in range(len(araim_outliers))]
    going_out = {}

    if true_outliers is not None:
        assert len(true_outliers) == len(araim_outliers), "ARAIM and truth need to have the same number of results"

    for key in results.keys():
        if key != 'ARAIM' and key != 'L2' and key != 'truth': 
            print(key)
            fgo_outliers = results[key][2] # Third element in the list
            assert len(fgo_outliers) == len(araim_outliers), "ARAIM and FGO methods need to have the same number of results"
            # Take the output and convert into sets of outliers, using a threshold
            threshold = thresholds.get(key,.1)
            print("Using threshold of", threshold)
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
    # - 5 = one outlier
    # - 6 = two outliers
    # - 7 = three outliers
    # - 8 = four outliers
    # - 10 = no outliers, no noise (for debugging)
    run_id = 5
    base_name = 'OneOutlier'
    results_file = base_name + '_results.pkl'
    errors_file = base_name + '_errors.pkl'
    # Get the true outliers from the database for comparison
    conn = sqlite3.connect('meas_data.db')
    snapshots = mdu.get_mc_sample_ids(conn, run_id)
    true_outliers = mdu.get_mc_samples_outliers(conn, snapshots)
    # Turn true_outliers into sets of indices
    true_outlier_sets = [set(np.where(true_outliers[i])[0].tolist()) for i in range(len(true_outliers))]
    results, errors = get_errors(results_file, errors_file)

    print('Keys for results are\n', results.keys())
    thresholds = {'GemanMcClure':.2, 'Huber':.4, 'Cauchy':.3}
    going_out = analyze_outliers(results, thresholds, true_outlier_sets)
    for key in going_out.keys():
        print(key)
        print(f'same {len(going_out[key][0])}, subset {len(going_out[key][1])}, superset {len(going_out[key][2])}, neither {len(going_out[key][3])}')

    # Let's look at some of the comparative sets between trunc_Gauss and ARAIM.  They should be the same, I think...
    araim_outliers = results['ARAIM'][2]
    fgo_outliers = results['gnc_trunc_Gauss'][2]
    print("Let's show 4 that are the same")
    max_cnt = 10
    curr_cnt = 0
    for same_idx in going_out['gnc_trunc_Gauss'][0]:
        print ('------------------- index: ', same_idx,'-------------------')
        print(f'gnc_trunc_Gauss: {np.where(fgo_outliers[same_idx] < .5)[0].tolist()}')
        print(f'ARAIM: {araim_outliers[same_idx]}')
        curr_cnt += 1
        if curr_cnt == max_cnt:
            break
    print('*********************************')
    print("Let's show 4 that are supersets of ARAIM")
    curr_cnt=0
    for super_idx in going_out['gnc_trunc_Gauss'][2]:   
        print ('------------------- index:', super_idx,'-------------------')
        print(f'gnc_trunc_Gauss: {np.where(fgo_outliers[super_idx] < .5)[0].tolist()}')
        print(f'ARAIM: {araim_outliers[super_idx]}')
        curr_cnt += 1
        if curr_cnt == max_cnt:
            break
    # print(going_out)
    