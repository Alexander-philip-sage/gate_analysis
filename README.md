# gate_analysis

## Current State

The data analysis for the paper was done using google-co-lab. The notebook that was used is included in this repo. The number of lines of code in the notebook got bigger than originally expected and it became less stable. The notebook would crash sometimes with its connection to the google vm. The data analysis itself was tested to be stably producing consistent results. To improve readability, professionalism, and maintainability we are currently migrating the notebook to this repo. The peak_valley_sim_stats.csv data differs from the csv created in the notebook. In the copying of functions between the envs a mistake must have been made. We have run out of time to continue debugging and testing this shift from the notebook and a python local run with checkpoints. A lot of work was put into this so it should be close

Checkpoints are important because the data loaded and created is way too large for RAM.

## Running the code

gate_analysis.py is the main entry point for the code. Check the functions called run_everything. There are several variations of this function that all do mostly the same thing. There is no one that is correct, it depends on what you need. 
