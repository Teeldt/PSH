# PSH
Code for the master thesis Pumped Storage Hydroelectricity for a Sustainable Electricity Transition, by Teodor Elmfeldt

# To run
## Dependencies
Code is written using Python 3.7.2
- with the numerical package NumPy,
- tables using pandas,
- regression using Statsmodels and
- plotting with Seaborn
- built upon Matplotlib.
Measuring data from SMHI was collected using smhi-open-data.

## Data
To run the code, there is need for supplemental data.

### Nordpool
Price data from Nordpool is needed, in the format of sundsek22.sdv, placed in the folder data/nordpool. Note that a ø needs to be manually removed from each file. This data is not available for open distribution, but is made available by Nordpool for academic purposes.

### Svenska Kraftnät
Electricity usage data from Svk can come from this source: https://www.svk.se/om-kraftsystemet/kraftsystemdata/elstatistik/, the used tables are 'Förbrukning och tillförsel per timme (i normaltid)' placed in data/svk and reformatted to .csv

FCR prices are from https://mimer.svk.se/ProductionConsumption/ProductionIndex and placed in the folder data/fcr as csv file

### SMHI
APIs are used to download relevant data, format it, and cache it, at runtime.

## Instructions
- operations_model.py contains the models used for simulation
- earnings_plots.py provides code for generating the plots regarding income from PSH operations
- stats_plots.py provides code for generating the plots regarding market conditions and other electricity statistics
- loader.py loads all dataframes from raw data
