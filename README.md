This code was used to create [this paper](visualizations/RAIM_and_RCF_comparison.pdf) that was published in the Institute of Navigation's ITM 2026 conference.  The presentation slides are [here](visualizations/ION_ITM_presentation.pdf), and a video recording of the presentation is [here](https://vimeo.com/1158474111?share=copy).

# Short code overview
If you want to use this code to generate results, you basically follow 4 steps (the first one is optional).
1. Create a database with epochs in it.  Or, just use the `meas_data.db` file that is part of the repository (using `git lfs` BTW, which may require some extra steps for you to get it onto your machine)
2. Create simulation runs using something like `mc_create_example.py`.  Keep track of which "run" you have made.  Each "run" is supposed to be different probabilities, outlier characteristics, etc.
3. Run all the RCF and the SS-ARAIM algorithm on the data.  I use `test_estimation.py`.  You will need to set the run you want to analyze and a name for the output data files
4. I use `analyze_results.py` to pull in the output data files and compute various metrics of interest.


# Code description
The heart of this code is a database setup as described in [data_structure.md](data_structure.md).  This database has all the information for the epochs that will be analyzed.  There are several functions for accessing this database in [meas_db_utils.py](meas_db_utils.py), including a function for creating the database, `create_measurement_database`.  (In other words, `create_measurement_database` is the formal specification of what I attempted to describe in [data_structure.md](data_structure.md).)  An example for creating the database is in [create_meas_database.py](create_meas_database.py).  Currently this uses datasets available at the [libRSF github site](https://github.com/TUC-ProAut/libRSF/tree/master/datasets/). The basic information from all the epochs (the real data, no simulated data) should be in this repository at `meas_data.db'.  Using that may be the easiest was to get started.

The idea is to have real satellite locations and ground-truth (receiver) locations from real data collects.  We can then use the real measurements, or, create our own simulated measurements with known noise characteristics. To see how to create your own Monte Carlo samples, you can see [mc_create_example.py](mc_create_example.py).  To see how to process data, see [L2_estimation_example.py](L2_estimation_example.py) and/or [test_estimation.py](test_estimation.py).  Processing data can also use the utilities in [comp_utils.py](comp_utils.py).  

# Real data processing
To process the real data, I do _not_ use the defaults for SS-RAIM.  Among other things, RAIM assumes dual-frequency data, and I don't believe the libRSF is this.  So I instead computed what the standard deviations was for all the pseudo-range measurements using a robust techinque that google gemini gave me.  (Robust because I want the standard deviation of just the "inliers" and not including the outliers.  It basically finds the median of the errors and multiplies that by a magic number to estimate the standard deviation.  The code for this is in [eval_pr_sd.py](eval_pr_sd.py).

## Standard Deviation of Pseudoranges
For reference, this is what we computed the sigma to be on the pseudo-ranges:
```
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
```

On a couple of datasets, I compared the performance using 14.4 vs. the standard deviation for that specific dataset, and it did not seem to make a significant difference, so I just use 14.4m throughout the paper.
