# ARAIM
- ARAIM, when computing residuals, uses a "dryTropo" model.  I removed this for simulation, which makes it as good in simulation.  With real data, ARAIM was much better than my methods.  Should try to move the dryTropo stuff into my methods and see how it does with real data!  Original was in `__determineObservedMinusComputed`

# With no outliers

Total Error (Magnitude) Statistics:
              ARAIM :   Average:   6.433 meters,  Std Dev:   3.543 meters,  Max:  24.571 meters
              Huber :   Average:   6.847 meters,  Std Dev:   3.944 meters,  Max:  28.205 meters
             Cauchy :   Average:   6.914 meters,  Std Dev:   4.005 meters,  Max:  29.770 meters
       GemanMcClure :   Average:   7.378 meters,  Std Dev:   4.404 meters,  Max:  34.188 meters
    gnc_trunc_Gauss :   Average:   6.521 meters,  Std Dev:   3.724 meters,  Max:  29.667 meters
   gnc_GemanMcClure :   Average:   6.978 meters,  Std Dev:   4.041 meters,  Max:  30.140 meters
                 L2 :   Average:   6.433 meters,  Std Dev:   3.543 meters,  Max:  24.572 meters

# With one outlier
Total Error (Magnitude) Statistics:
              ARAIM :   Average:   7.274 meters,  Std Dev:   4.752 meters,  Max:  51.113 meters
              Huber :   Average:   7.636 meters,  Std Dev:   9.619 meters,  Max: 242.090 meters
             Cauchy :   Average:   7.507 meters,  Std Dev:   6.587 meters,  Max: 161.380 meters
       GemanMcClure :   Average:   8.075 meters,  Std Dev:   7.418 meters,  Max: 179.855 meters
    gnc_trunc_Gauss :   Average:   7.374 meters,  Std Dev:   9.967 meters,  Max: 243.375 meters
   gnc_GemanMcClure :   Average:   7.948 meters,  Std Dev:  10.000 meters,  Max: 243.330 meters
                 L2 :   Average:  28.307 meters,  Std Dev:  27.502 meters,  Max: 241.084 meters

# With two outliers
Total Error (Magnitude) Statistics:
              ARAIM :   Average:  10.086 meters,  Std Dev:  26.280 meters,  Max: 648.512 meters
              Huber :   Average:  10.063 meters,  Std Dev:  26.644 meters,  Max: 716.096 meters
             Cauchy :   Average:  10.373 meters,  Std Dev:  27.552 meters,  Max: 719.020 meters
       GemanMcClure :   Average:  11.826 meters,  Std Dev:  28.760 meters,  Max: 719.326 meters
    gnc_trunc_Gauss :   Average:   9.483 meters,  Std Dev:  26.641 meters,  Max: 719.340 meters
   gnc_GemanMcClure :   Average:  10.191 meters,  Std Dev:  24.905 meters,  Max: 642.766 meters
                 L2 :   Average:  42.170 meters,  Std Dev:  35.988 meters,  Max: 544.641 meters

# With four outliers
Total Error (Magnitude) Statistics:
              ARAIM :   Average:  34.115 meters,  Std Dev:  40.387 meters,  Max: 449.928 meters
              Huber :   Average:  20.442 meters,  Std Dev:  37.326 meters,  Max: 370.127 meters
             Cauchy :   Average:  21.675 meters,  Std Dev:  40.075 meters,  Max: 370.163 meters
       GemanMcClure :   Average:  26.328 meters,  Std Dev:  47.203 meters,  Max: 485.886 meters
    gnc_trunc_Gauss :   Average:  19.083 meters,  Std Dev:  35.127 meters,  Max: 354.737 meters
   gnc_GemanMcClure :   Average:  20.682 meters,  Std Dev:  38.976 meters,  Max: 371.437 meters
                 L2 :   Average:  59.840 meters,  Std Dev:  43.779 meters,  Max: 386.683 meters

# TODO
- Pull out cov from ARAIM and allow others to use VPL and HPL calculation
- Show that for no outliers, things are about the same for all methods
- Vary Gauss truncation cutoff like ARAIM does (low priority)