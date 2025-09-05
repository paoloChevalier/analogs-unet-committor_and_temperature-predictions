# analogs-unet-committor_and_temperature-predictions

Work on committor function estimation using analogs and temperature predictions using UNET networks in ERA5 to evaluate the transfer learning from IPSL-CM6A-LR. This work was done during my internship at LSCE (Laboratoire des Sciences du Climat et de l'Environnement) and the CEA (Commissariat à l'énergie atomique et aux énergies alternatives)

## Installation

To use locally, make sure you have the pangeo distribution installed for general scripts/notebooks and a distribution including tensorflow >2.15 for the scripts on UNETs.
All scripts and notebooks should work seamlessly on the IPSL mésocentre computing center (spirit(x) and hal).

## Usage

### Preprocess

Preprocessing scripts

- ERA5_preprocess.py : preprocess the data, encoding time as `year*10^4+month*10^2+day`
- IPSL_CM6A_LR_preprocess.py : same, also merges the data from the 33 IPSL-CM6A-LR runs, encoding time as `run*10^8+year*10^4+month*10^2+day`
- train_val_sets.py : separates both datasets as 80% training, 20% validation

### Analogs 

For best performance, use on machines with >120gb ram, if you don't have enough ram you can always play with the chunk sizes to prevent your ram from running out.

- compute_analogs.py : necessary functions to compute analogs and some computations.

### UNETs

Tensorflow is most efficient with GPUs, make sure you have one available.

-
-
-
-

### Figures and notebooks

All the figures in my report were produced with the notebooks in this repo

- ERA5_exploration.ipynb : exploration of the data from ERA5 (reanalysis)
- IPSL_CM6A_LR_exploration.ipynb : exploration of the data from IPSL-CM6A-LR (simulation)
- compare_rmse_compute.ipynb : tests in order to compare code for computing RMSE
- results.ipynb : analysing results from analogs and ML
- climTools/helperFunctions.py : local package with useful functions I used often

## Data

The data needed is from IPSL-CM6A-LR and ERA5. Both available online [here](https://esgf-node.ipsl.upmc.fr/search/cmip6-ipsl/) and [here](https://cds.climate.copernicus.eu/).

The models and analogs are available [here]()

## Contributing

As this work was done during an internship, I probably won't have the time to update this repo nor deal with contributions if not small. If needed I am always reachable by email preferably.

## License
This work is the property of the Commissariat à l'énergie atomique et aux énergies alternatives.
