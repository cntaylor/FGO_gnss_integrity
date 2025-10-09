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
3.  A table of monte carlo parameters.  This should be a fairly small table, but specifies (records) how data was generated.  First ID will be "real data", 2nd ID will be the number and then a JSON with parameters to specify how each pseudorange will be modified.
4.  A table with unique Monte Carlo ID, which Dataset + snapshot_num it is applied to, and a link to which Monte Carlo parameters are being used
5.  A table of measurements.  It will have the Monte Carlo ID, followed by a satellite number and a pseduorange value.
6.  A table of methods used to determine position and raise alerts.  The 6th table is similar to the 3rd table and basically stores method_IDs and parameters for the methods which are used to generate results.
7.  TODO:  Has a Monte Carlo ID, a Method ID, and then results.  Results should include error (x,y,z, Horz, vert, and 3D), protection level (all levels?), and alerts?  Do we also want which satellites were excluded?  Need to decide on this


Note that there is some special stuff (a VIEW) needed to link the measurements in table 5 to the satellites in table 2, when 5 only has a key to table 4, which links to table 2.