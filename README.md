# loss_capacity


### install dependencies
install liftoff for managing experiments:
pip install git+https://github.com/tudor-berariu/liftoff.git#egg=liftoff

###For listoff usage: 

start a simple python command with the flags given by the file ...default_config_dev.yaml
this is just for starting a single experiment with output given in the terminal
liftoff indep_penalty_5L_config_single_seed.py config/default_config_dev.yaml

create a list of experiments according to the files in config/experiment1
liftoff-prepare ./config/experiment1/ --runs-no 1 --results-path ./results --do


#
gpu 0 1 2 3 means use 4 gpus
-per-gpu 8 means that for each gpu you start 8 scripts
--procs-no 32 means that you have 32 experiments in total, running in parallel on the same machine (on cluster)
liftoff IndepMechanism/indep_penalty_5L_config.py IndepMechanism/results/2022Jul10-174624_syn_3L_corr_penalty --gpus 0 1 2 3 --per-gpu 8 --procs-no 32 --results-path IndepMechanism/results

liftoff-clean path --crashed-only --clean-all --do # delete all files on crashed exp

you can run without --do to see what files it will delete

--crashed-only  tells it to delete only the runs that have an error

--clean-all tells it to delete all produced files in the folders (e.g. logs/ model_files etc)

### how to train
The main training file is src/train_unsupervised_model.py
to submit it in the cluster, use configs/... yaml files
