source /home/anicolicioiu/miniconda3/etc/profile.d/conda.sh
conda activate capacity_env_cuda11
module load cuda/11.3
#cd /home/anicolicioiu//projects/loss_capacity/src/ 


#OMP_NUM_THREADS=1 liftoff train_probe.py  ./results/2022Jun02-204408_round_10_vae_probes/  --gpus 0 --per-gpu 8 --procs-no 8
#OMP_NUM_THREADS=1 liftoff train_probe.py  ./results/2022Jun02-204634_round_10_vae_probes  --gpus 0 --per-gpu 8 --procs-no 8
#OMP_NUM_THREADS=1 liftoff train_probe.py ./results/2022Jun02-122943_round_10_unif_small_mult  --gpus 0 --per-gpu 8 --procs-no 8
#OMP_NUM_THREADS=1 liftoff train_probe.py ./results/2022Jun01-115933_round_10_unif/  --gpus 0 --per-gpu 3 --procs-no 3


#liftoff train_probe.py  ./results/2022Jun05-140441_round_10_rf_vae_beta_0_100  --gpus  --per-gpu 1 --procs-no 1
#liftoff train_probe.py  ./results/2022Jun04-124219_round_10_rf_unif_noise/  --gpus  --per-gpu 1 --procs-no 1
#liftoff train_probe.py  ./results/results_plot1_rf/vae_beta1/  --gpus  --per-gpu 1 --procs-no 1
#liftoff train_probe.py  ./results/2022Jun04-124447_round_10_rf_raw//  --gpus  --per-gpu 1 --procs-no 1
#liftoff train_probe.py  ./results/2022Jun04-124644_round_10_rf_resnet/  --gpus  --per-gpu 1 --procs-no 1



#OMP_NUM_THREADS=1 liftoff train_probe.py ./results/2022Jun03-234445_round_11_resnet/  --gpus 0 --per-gpu 2 --procs-no 2
#OMP_NUM_THREADS=1 liftoff train_probe.py ./results/2022Jun04-185745_round_11_raw/  --gpus 0 --per-gpu 1 --procs-no 1
#OMP_NUM_THREADS=1 liftoff train_probe.py ./results/2022Jun04-002752_round_11_vae_probes  --gpus 0 --per-gpu 3 --procs-no 3
#OMP_NUM_THREADS=1 liftoff train_probe.py ./results/2022Jun04-190644_round_11_unif_noisy  --gpus 0 --per-gpu 6 --procs-no 6


#OMP_NUM_THREADS=1 liftoff train_probe.py ./results/2022Jun04-120556_round_11_resnet/  --gpus 0 --per-gpu 2 --procs-no 2
#OMP_NUM_THREADS=1 liftoff train_probe.py ./results/2022Jun04-121024_round_11_raw/  --gpus 0 --per-gpu 1 --procs-no 1
#OMP_NUM_THREADS=1 liftoff train_probe.py ./results/2022Jun04-002752_round_11_vae_probes  --gpus 0 --per-gpu 6 --procs-no 6
#OMP_NUM_THREADS=1 liftoff train_probe.py ./results/2022Jun06-011130_round_11_unif_noisy  --gpus 0 --per-gpu 6 --procs-no 6




#OMP_NUM_THREADS=1 liftoff train_probe.py results_mpi3d/results_mpi3d_rff/2022Jun28-151441_round_12_vae_probes_rff_const_g  --gpus 0 --per-gpu 8 --procs-no 8 --results-path results_mpi3d/results_mpi3d_rff


OMP_NUM_THREADS=1 liftoff train_probe.py results_mpi3d/results_mpi3d_rff/2022Jun28-150647_round_12_resnet_rff2  --gpus 0 --per-gpu 4 --procs-no 4 --results-path results_mpi3d/results_mpi3d_rff
OMP_NUM_THREADS=1 liftoff train_probe.py results_mpi3d/results_mpi3d_rff/2022Jun28-150707_round_12_resnet_rff2  --gpus 0 --per-gpu 4 --procs-no 4 --results-path results_mpi3d/results_mpi3d_rff
OMP_NUM_THREADS=1 liftoff train_probe.py results_mpi3d/results_mpi3d_rff/2022Jun28-150714_round_12_resnet_rff2  --gpus 0 --per-gpu 4 --procs-no 4 --results-path results_mpi3d/results_mpi3d_rff


#OMP_NUM_THREADS=1 liftoff train_probe.py results_mpi3d/results_mpi3d_rff/2022Jun28-152303_round_12_raw_rff  --gpus 0 --per-gpu 2 --procs-no 2 --results-path results_mpi3d/results_mpi3d_rff

#OMP_NUM_THREADS=1 liftoff train_probe.py results_mpi3d/results_mpi3d_rff/2022Jun28-153039_round_12_unif_noisy_rff/  --gpus 0 --per-gpu 8 --procs-no 8 --results-path results_mpi3d/results_mpi3d_rff


## CARS3D




# mlps
#OMP_NUM_THREADS=1 liftoff train_unsupervised_model.py  ./results/2022Jun21-203831_unsup_vae_cars3d/  --gpus 0 --per-gpu 1 --procs-no 1 --results-path results_cars3d
#OMP_NUM_THREADS=1 liftoff train_probe.py   ./results/2022Jun26-153217_round_11_resnet_mlp185/  --gpus 0 --per-gpu 1 --procs-no 1
#OMP_NUM_THREADS=1 liftoff train_probe.py ./results/2022Jun26-155807_round_11_raw_mlp/  --gpus 0 --per-gpu 3 --procs-no 3
#OMP_NUM_THREADS=1 liftoff train_probe.py ./results/2022Jun24-150243_round_11_vae_probes_mlp  --gpus 0 --per-gpu 8 --procs-no 8
#OMP_NUM_THREADS=1 liftoff train_probe.py  ./results/2022Jun23-154623_round_11_unif_noisy_mlp/  --gpus 0 --per-gpu 8 --procs-no 8

#rf
#OMP_NUM_THREADS=1 liftoff train_probe.py  ./results/2022Jun22-205153_round_10_rf_raw/  --gpus --per-gpu 1 --procs-no 1
#OMP_NUM_THREADS=1 liftoff train_probe.py  ./results/2022Jun22-203854_round_10_rf_resnet/  --gpus --per-gpu 1 --procs-no 1
#OMP_NUM_THREADS=1 liftoff train_probe.py  ./results/2022Jun23-143148_round_10_rf_vae_beta  --gpus --per-gpu 1 --procs-no 1

#rff
#OMP_NUM_THREADS=1 liftoff train_probe.py  ./results/2022Jun23-163807_round_12_resnet_rff/  --gpus 0 --per-gpu 7 --procs-no 7
#OMP_NUM_THREADS=1 liftoff train_probe.py ./results/2022Jun23-163856_round_12_raw_rff  --gpus 0 --per-gpu 7 --procs-no 7
#OMP_NUM_THREADS=1 liftoff train_probe.py ./results/2022Jun24-144145_round_12_vae_100_probes_rff  --gpus 0 --per-gpu 7 --procs-no 7
#OMP_NUM_THREADS=1 liftoff train_probe.py  ./results/2022Jun23-163628_round_12_unif_noisy_rff  --gpus 0 --per-gpu 7 --procs-no 7


# shapes3d


#OMP_NUM_THREADS=1 liftoff train_unsupervised_model.py ./results/2022Jun20-142407_unsup_vae_cars3d/  --gpus 0 --per-gpu 1 --procs-no 1
#OMP_NUM_THREADS=1 liftoff train_probe.py ./results/2022Jun14-231042_round_12_vae_probes_rff/  --gpus 0 --per-gpu 8 --procs-no 8
#OMP_NUM_THREADS=1 liftoff train_probe.py ./results/2022Jun20-144831_round_11_raw  --gpus 0 --per-gpu 2 --procs-no 2
#OMP_NUM_THREADS=1 liftoff train_probe.py ./results/2022Jun20-144631_round_11_resnet  --gpus 0 --per-gpu 2 --procs-no 2
#OMP_NUM_THREADS=1 liftoff train_probe.py ./results/2022Jun15-115930_round_12_unif_noisy_rff/  --gpus 0 --per-gpu 6 --procs-no 6

# DSPRITES

# mlps
#OMP_NUM_THREADS=1 liftoff train_unsupervised_model.py results_dsprites/2022Jun27-001452_unsup_vae_dsprites/  --gpus 0 --per-gpu 1 --procs-no 1 --results-path results_dsprites
#OMP_NUM_THREADS=1 liftoff train_probe.py  results_dsprites/2022Jun26-003504_round_11_raw_resnet_mlp/  --gpus 0 --per-gpu 5 --procs-no 5 --results-path results_dsprites
#OMP_NUM_THREADS=1 liftoff train_probe.py ./results/2022Jun24-145925_round_11_raw_mlp  --gpus 0 --per-gpu 7 --procs-no 7
#OMP_NUM_THREADS=1 liftoff train_probe.py results_dsprites/2022Jun26-221918_round_11_vae_probes_mlp/  --gpus 0 --per-gpu 3 --procs-no 3 --results-path results_dsprites
#OMP_NUM_THREADS=1 liftoff train_probe.py  ./results/2022Jun23-154623_round_11_unif_noisy_mlp/  --gpus 0 --per-gpu 8 --procs-no 8


# rf
#liftoff train_probe.py  results_dsprites/2022Jun26-150104_round_10_rf_raw_resnet  --gpus  --per-gpu 1 --procs-no 1 --results-path results_dsprites
#OMP_NUM_THREADS=1 liftoff train_probe.py  results_dsprites/2022Jun26-145505_round_10_rf_raw_resnet/  --gpus  --per-gpu 1 --procs-no 1 --results-path results_dsprites


# RFF

#OMP_NUM_THREADS=1 liftoff train_probe.py  results_dsprites/2022Jun26-215706_round_12_vae_rff/  --gpus 0 --per-gpu 8 --procs-no 8 --results-path results_dsprites
#OMP_NUM_THREADS=1 liftoff train_probe.py   results_dsprites/2022Jun26-215908_round_12_raw_resnet_rff/  --gpus 0 --per-gpu 5 --procs-no 5 --results-path results_dsprites
