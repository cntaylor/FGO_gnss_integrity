The heart of this code is a database setup as described in [data_structure.md](data_structure.md) and setup in [setup_db.py](setup_db.py).  This database stores satellite .  There are several functions for accessing this database in [meas_db_utils.py](meas_db_utils.py), including a function for creating the database, `create_measurement_database`.  (In other words, `create_measurement_database` is the formal specification of what I attempted to describe in [data_structure.md](data_structure.md).)  An example for creating the database is in [create_meas_database.py](create_meas_database.py).  Currently this uses datasets available at the [libRSF github site](https://github.com/TUC-ProAut/libRSF/tree/master/datasets/).  

The idea is to have real satellite locations and ground-truth (receiver) locations from real data collects.  We can then use the real measurements, or, create our own simulated measurements with known noise characteristics. To see how to create your own Monte Carlo samples, you can see [mc_create_example.py](mc_create_example.py).  To see how to process data, see [L2_estimation_example.py](L2_estimation_example.py).  Processing data can also use the utilities in [comp_utils.py](comp_utils.py).  

## Standard Deviation of Pseudoranges
For reference, this is what we computed the sigma to be on the pseudo-ranges:
'Chemnitz': 16.31049057163418, 
'UrbanNav_Deep': 10.490736627831403, 
'UrbanNav_Harsh': 17.156413363268225, 
'UrbanNav_Medium': 10.923237852979847, 
'UrbanNav_Obaida': 10.177368156231008, 
'UrbanNav_Shinjuku': 11.52246230831961, 
'Frankfurt_Westend': 16.15050987864886, 
'Frankfurt_Main': 30.61345085121137, 
'Berlin_Gendarmenmarkt': 18.441702702800132, 
'Berlin_Potsdamer': 22.42617181442883, 

'combined': 14.428315616394636

Some results:  Using Huber (and 14.4 as the std dev) shows a significant improvement in localization on Chemnitz (avg err 65.9m -> 32.4.)  

On Berlin-postdamer, however, results not nearly as impressive (77->66m, Huber; 77->65, Cauchy). Changing std dev to 22.4 did not improve results (77->69m)

On Chemnitz, moving to GemanMcClure made little difference (65.9 -> 31.3)
Cauchy : (65.9 -> 30.1)