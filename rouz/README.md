# Info

This is my (Rouz) playground for some time series analysis for 
electricity price forecasting. The data is sourced from the 
NRG website as well as some weather and gas price data that 
Nic scraped. 

Two data sets are used for the work in this folder. Both use the 
same underlying raw data but differ in the constructed feature 
sets.

## Directories:
The git-directories are 
    1. `src/`: the source directory, containing the Jupyter notebooks
and the Python scripts. I followed the TensorFlow tutorial page on time 
series forecasting available [here](https://www.tensorflow.org/tutorials/structured_data/time_series#setup).
    2. `plots/`: PDF plots, contains two types of plots, each in two 
sets
    ** `plots/dat_set_{rouz,nic}`: example plots of three time windows
and the resulting model predictions. The examples are from the training 
sets of the specified data set. Each folder contains 2 subfolders for 
single and multi-step predictions (1hr into the future vs 24 hours).
    ** `plots/scores_{Rouz,Nic}_data`: contains the test and validation
score plots (RMSE).
    3. `correlations/`: three `.csv` files containing the correlations
of the various features from the Rouz dataset, only the training set 
was used to compute these.
