# !/bin/sh
. /home/wliang/miniconda3/etc/profile.d/conda.sh
conda activate vae

echo $CONDA_PREFIX
# module load cuda/11.3/ --gpus 0 1 2 3 --per-gpu 8 --procs-no 32 --results-path results
module load cuda/11.3
python --version
cd /is/ei/wliang/loss_capacity/src/
liftoff train_unsupervised_model.py ./results/2022Sep08-214017_unsup_vae_dsprites_hsicbetavae/ --gpus 0 --per-gpu 1 --procs-no 1 --results-path results
