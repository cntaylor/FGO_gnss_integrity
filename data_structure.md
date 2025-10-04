```
Dataset

|-- Snapshot (unknown # per dataset)
|
|  |-- Time
|  |-- True Location (x,y,z, in ECEF)
|  |-- (N times) Satellite location (in ECEF)
|  |-- Monte Carlo Iteration
|  |  |-- Real or simulated (bool)
|  |  |-- (N times) Pseudorange  
|  |  |-- Results (unknown # per Monte Carlo iteration)
|  |  |  |-- Technique
|  |  |  |-- Parameters used
|  |  |  |-- Fault declared?
|  |  |  |-- Protection level
|  |  |  |-- Estimated location
```