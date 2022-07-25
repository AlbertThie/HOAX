
Hyperparameter Optimisation Algorithm Explorer
===============================

HOAX can be used to optimise the hyperparameters of machine learning algorithms in quantum chemistry.


Instructions
--------
Installation
************

Install via pip

```
pip install hoax
```
Launch from the terminal, call hoax with the database file and config file as arguments.

```
hoax ["databasefile"] ["configfile"] (run in terminal)
```

The HOAX process will run until completion, the duration depends on the settings in the config file. After completion, the hoax package will provide a database with the overview of the optimization process and the best performing neural network. The names of these files can be set in the config file, under *logging_file* and *model_filename* respectively. The optimization process file contains the scanned parameters and validation errors of all runs that were attempted by HOAX, in the HDF5 format. The best performing neural network can be used with the validator to run test trajectories and is provided as a pytorch network.


Config file
**************

The config file contains the metadata from the software package that was used to generate it, though this is optional.
The config file also contains the setup parameters for the database, neural network and chosen hyperparameter optimization.


```
{"config": {
    "debug": "on",
    "database": { 
        "file" : "db.dat",
        "properties": [
        "energy",
        "gradient"
            ],
        "crdmode": "cartesian",
        "timestamp": "2021-04-23T18:25:43.511Z",
        "mode": "ab-initio",
        "ab-initio": {
            "exe" : "QChem.exe",
            "chg" : 0,
            "mult": 1,
            "exchange" : "pbe0",
            "basis" : "cc-pvdz",
            "max_scf_cycles" : 500,
            "xc_grid" : "000075000302",
            "mem_static" : 4000,
            "mem_total" : 16000,
            "sym_ignore" : true,
            "set_iter" : 50,
            "input_bohr" : true
        }
 
    },
    "neural_network": {

        "epochs" : 1000,
        "epoch_step"  : 50,
        "logging_file" : "logfile.txt",
        "optimizer": "adam",
        "activation" : "tanh",
		"loss_function" : "MSE",
        "model_filename" : "Network.pt"
		
            },
	"grid_search":{
		"validation_ratio": 0.1,
		 "hiddenlayer_size" : [10,100,10],
		"hiddenlayer_number": [1,10,1],
        "learning_rates" : [1,0.1,0.01,0.001,0.0001]
	}	
			
}}

```

Optimization output file
**************








