Central to much of this code is an SQL database.  This database has two main purposes:
1. Store "epochs" with all the information needed to localize. 
2. Store results of processing the epoch information by different "methods" so that comparisons can be made in terms of outlier rejection and performance  (not really implemented at all yet.)

# Epoch information
For each epoch, the following information should be available:
  - The true location of the receiver
  - The locations of all the satellites
  - The pseudo-range measurements from the satellite to the receiver
Note that the goal of this project is to use Monte Carlo simulations to test out epochs with lots of different outliers and possible problems.  Therefore, each "epoch" is divided into two portions.  The first is the true location of the receiver and the locations of all the satellites.  The second portion is a Monte Carlo sample of what the pseudoranges will be.  There will be multiple sets of pseudoranges per "epoch" representing possible errors on the pseduoranges.

# Result information
The goal of this project is to test out algorithms that accept a set of pseudorange information (with the associated satellite locations) and estimation a location.  Furthermore, it should provide some integrity information. 

# Database setup
To enable a setup that records everything listed above, we will use a SQL database with 7 interacting tables.  Note that often there are two or three tables that are really used to store a single piece of information, but by splitting out the tables, data storage will be far more efficient.  At a high level:
  - Tables 1 & 2 store the first portion of the epoch information for all the epochs.  This provides sufficient information to generate noiseless pseudo-range estimates
  - Tables 3-5 are used to store simulated data, 
  - Table 6-7 are used to store results.

In more detail (and for exact detail, see [setup_db.py](setup_db.py))
1.  This table stores information for each timestep.  Each entry in the table will have a Sanpshot_ID, what dataset it came from, what time is associated with that epoch, and the true location of the receiver.  Entries are indexed by "Sanpshot_ID".
2.  A table for each satellite.  This will have the epoch_ID (from Table 1) + a satellite num (starting at 0 for each epochID), and location of that satellite for that epoch.
3.  A table of monte carlo runs.  This should be a fairly small table, but specifies (records) how data was generated.  MC_run_ID = 1 will be for "real data".  Later Run IDs will stand for simulated data and should record what parameters were used to generate the data.  Note that you could have multiple rows with the same parameters (if you run the same MC approaches multiple times).
4.  A table with unique Monte Carlo ID for each sample (really, a table of all samples ever created).  For each sample, it records which epoch_ID  and MC_Run_ID's it pertains to.
5.  A table of measurements.  It will have the Monte Carlo sample ID, followed by a satellite number and a pseduorange value.  Also has some values for (from simulated data) if it was an outlier (to evaluate how effective fault detection is).  Outlier information can be NULL, meaning we don't know (like with real data).
6.  A table of methods used to determine position and raise alerts.  The 6th table is similar to the 3rd table and basically stores method_IDs and parameters for the methods which are used to generate results.
7.  Has a Monte Carlo Sample ID, a Method ID, and then results.  Results should include error (location estimate - truth, x,y,z, in ECEF about truth), and two follow-ons: a JSON data/metadata field and a BLOB data field, that will store all results for that method.  This is a very generic field and may be modified as we flesh out the methods we are using and what they will return.


# Main operations
Let's think through two or three main things that will need to be done and then write functions to do this:

(Mark things with a (db) if it is a database interaction)

## Create a bunch of (Monte Carlo) samples
Will need to:
* (db) Get a list of all possible epoch_IDs -- get_epoch_ids (optional dataset name parameter)
* Select which epoch_IDs will be used to generate data
* (db) Get all epoch information -- epoch_data(epoch_ids)
* Create synthetic measurements for each epoch
* (db) Store all the MC samples back in the database -- add_MC_samples (list of tuples with epoch_ID and 1D numpy array of pseudoranges)

Functions for doing all of these steps are in [meas_db_utils.py](meas_db_utils.py) and an example of using these functions can be found int [mc_create_example.py](mc_create_example.py)

## Process data for a set of MC samples
If we want to process a bunch of data samples and store results in the table, we will need to:
* (db?) Get the MC_sample IDs you want to process (assume is done for a particular MC_run) -- get_MC_sample_IDs (MC_run_ID, optional dataset)
* (db) Get the measurements for each MC sample -- get_MC_sample_measurements
* Run some algorithm to process the measurements
* (db-unknown) Store the results in the DB
* (db) Get the truth for each MC sample -- get_MC_sample_truths
* Use the truth to generate results

A first example can be found in [L2_estimation_example.py](L2_estimation_example.py), though it does not store any results in the database.  A more advanced example will be in [FG_estimation.py](FG_estimation.py)