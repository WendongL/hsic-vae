source /home/anicolicioiu/miniconda3/etc/profile.d/conda.sh
conda activate capacity_env
cd /home/anicolicioiu//projects/loss_capacity/src/ 
liftoff train_supervised_model.py ../configs/default_config.yaml
