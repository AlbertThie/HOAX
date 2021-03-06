
Hyperparameter Optimisation Algorithm Explorer
===============================

HOAX can be used to optimise the hyperparameters of machine learning algorithms in quantum chemistry.


Instructions
--------
Installation
************

Install via pip

:code:`pip install hoax` 

Launch from the terminal, call hoax with the database file and config file as arguments.

:code:`hoax ["databasefile"] ["configfile"] (run in terminal)

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
        "plotting_file" : "plottingfile.txt",
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


