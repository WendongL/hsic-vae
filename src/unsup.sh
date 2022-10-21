# !/bin/sh
. /home/wliang/miniconda3/etc/profile.d/conda.sh
conda activate vae1

echo $CONDA_PREFIX
# module load cuda/11.3/ --gpus 0 1 2 3 --per-gpu 8 --procs-no 32 --results-path results
# module load cuda/11.3
python --version
cd /home/wliang/Github/loss_capacity/src/
liftoff train_unsupervised_model.py ./results/Oct21/dsprites/2022Oct21-083112_dsprites_hsicbetavae_notonlyhsic/ --gpus 0 --per-gpu 1 --procs-no 1 --max-runs 1 --results-path results
