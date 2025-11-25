import numpy as np
import matplotlib.pyplot as plt
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

default_params = {
    "rcf" : "Huber",  # Robust cost function
    "base_sigma" : 14.4,  # Base measurement noise standard deviation (meters)
    "gnc" : False,    # Graduated non-convexity
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
    
def snapshot_fgo(measurements, params = default_params):
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
        # # Use the Li-Ta approach of setting the original "weight" high enough that all residuals are in the 
        # # convex region of the robust cost function, then gradually reduce it.
        # # Find the maximum residual from the initial estimate
        # if params["rcf"] not in ("GemanMcClure", "Geman-McClure", "Geman_McClure"):
        #     raise ValueError("GNC is only implemented for Geman-McClure RCF in this example.")
        # y = np.zeros(len(measurements))
        # for i,meas in enumerate(measurements):
        #     est_pseudorange = mdu.noiseless_pseudorange(est_location, meas[1:4])
        #     y[i] = meas[0] - (est_pseudorange + time_offset)
        # max_residual = np.max(np.abs(y))/params['base_sigma']
        # # Equation 23 from Li-Ta paper
        # theta = 3 * max_residual**2 / params.get("geman_c", 2.0)**2
        # # To perform GNC, will have two loops.  Outer loop reduces theta, inner loop does IRLS
        # while theta > 1.:
        #     # Inner loop: Perform IRLS
        #     weights = get_rcf_weights(y, params)
        #     # Update residuals
        #     for i,meas in enumerate(measurements):
        #         est_pseudorange = mdu.noiseless_pseudorange(est_location, meas[1:4])
        #         y[i] = meas[0] - (est_pseudorange + time_offset)
        #     max_residual = np.max(np.abs(y))/params['base_sigma']
        #     theta = 3 * max_residual**2 / params.get("geman_c", 2.0)**2
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
                print("Problem with robust FGO ... No longer of sufficient rank!  Quitting prematurely")
                delta_mag = 0.
            num_iters += 1
    #Compute the covariance
    # First, need lat/lon
    lat_lon = r3f.ecef_to_geodetic(est_location)
    # Then the rotation from ECEF to ENU
    C_n_ecef = r3f.dcm_ecef_to_navigation( lat_lon[0], lat_lon[1], ned=False)
    rot_mat = np.zeros((4,4))
    rot_mat[:3,:3] = C_n_ecef
    rot_mat[3,3] = 1
    return est_location, time_offset, weights, rot_mat@np.linalg.inv(Jp.T@Jp)@rot_mat.T, num_iters

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

