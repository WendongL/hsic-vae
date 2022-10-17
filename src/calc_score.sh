# !/bin/sh
. /home/wliang/miniconda3/etc/profile.d/conda.sh
conda activate vae

echo $CONDA_PREFIX
# module load cuda/11.3/ --gpus 0 1 2 3 --per-gpu 8 --procs-no 32 --results-path results
module load cuda/11.3
python --version
cd /is/ei/wliang/loss_capacity/src/
liftoff calc_HSIC.py ./results/2022Sep12-120112_unsup_vae_dsprites_calc_betavae --gpus 0 --per-gpu 2 --procs-no 2 --results-path results
