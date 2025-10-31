I am going to create a SQL database that includes all data that may be used to test different integrity settings.  The data will all be based off of true data collects.  It should look something like the following.

```
Dataset

|-- Snapshot (unknown # per dataset)
|
|  |-- Time
|  |-- True Location (x,y,z, in ECEF)
|  |-- (N times) Satellite location (in ECEF)
|  |-- Monte Carlo Iteration
|  |  |-- MC params (including sim vs. real)
|  |  |-- (N times) Pseudorange  
```

The difficulty with this dataset is having long lists of unknown length.  Therefore, we will use a SQL database with 7 interacting tables.  Tables 1 & 2 can be used to generate noiseless pseudo-range estimates.  Tables 3-5 are used to store simulated data so every simulation scenario is stored.  The 6th and 7th table will be used to store results.
1.  The table storing information for each timestep.  Each entry in the table will have Dataset_ID, Snapshot_num, the time, and the true location.  Dataset_ID + snapshot_num will be the key to finding every specific entry.
2.  A table for each satellite.  This will have the Dataset_ID + snapshot_num, satellite_num (starting at 1 for each main key), and location of the satellite
3.  A table of monte carlo runs.  This should be a fairly small table, but specifies (records) how data was generated.  First ID will be "real data", 2nd ID will be the number and then a JSON with parameters to specify how each pseudorange will be modified.  Note that you could have multiple rows with the same parameters (if you run the same MC approaches multiple times).
4.  A table with unique Monte Carlo ID for each sample (really, a table of all samples ever created), which Dataset + snapshot_num it is applied to, and a link to which Monte Carlo run the sample belongs to.
5.  A table of measurements.  It will have the Monte Carlo sample ID, followed by a satellite number and a pseduorange value.  Also has some values for (from simulated data) if it was an outlier (to evaluate how effective fault detection is).  Outlier information can be NULL, meaning we don't know (like with real data).
6.  A table of methods used to determine position and raise alerts.  The 6th table is similar to the 3rd table and basically stores method_IDs and parameters for the methods which are used to generate results.
7.  TODO:  Has a Monte Carlo Sample ID, a Method ID, and then results.  Results should include error (location estimate - truth, x,y,z, in NED about truth), covariance (NED about truth in meters -- possible NULL), protection level?  whatever is needed to store ARAIM results (horizontal, verticle, and 3D), and satellites inlier/outlier selection (possible NULL)?


Design philosophy:  In many of these tables, there is a decision to make between having a compound key (e.g. DATASET_ID and snapshot_num together) which may be easier for the user to think about, but will also lead to larger tables.  I am instead opting for unique IDs (snapshot_ID) as that is smaller across multiple tables.  This means the IDs are just arbitrary numbers identifying them, but I believe you can pull the original data back out if needed and it makes things more efficient.


# Main operations
Let's think through two or three main things that will need to be done and then write functions to do this:

(Mark things with a (db) if it is a database interaction)

## Create a bunch of samples
Will need to:
* (db) Get a list of all possible snapshot_IDs -- get_snapshot_ids (optional dataset name parameter)
* Select which snapshot_IDs will be used to generate data
* (db) Get all snapshot information -- snapshot_data(snapshot_ids)
* Create synthetic measurements for each snapshot
* (db) Store all the MC samples back in the database -- add_MC_samples (list of tuples with snapshot_ID and 1D numpy array of pseudoranges)

## Process data for a set of MC samples
* (db?) Get the MC_sample IDs you want to process (assume is done for a particular MC_run) -- get_MC_sample_IDs (MC_run_ID, optional dataset)
* (db) Get the measurements for each MC sample -- get_MC_samples_meas
* Run some algorithm to process the measurements
* (db-unknown) Store the results in the DB
* (db) Get the truth for each MC sample -- get_MC_samples_truth
* Use the truth to generate results