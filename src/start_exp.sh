NAME=$1
MODEL=$2
LAYERS=$3
MULT=$4
SUPERVISION=$5
EPOCHS=$6
LR=$7
PRETRAINED=$8
NOISE_STD=$9
DATASET=${10}
echo "-------"
echo "${10}"
echo "-------"
echo '1_'$1'_2_'$2'_3_'$3'_4_'$4'_5_'$5'_6_'$6'_7_'$7'_8_'$8'_9_'$9'_10_'$DATASET
RAND=1
model_name=$1'_'$2'_'$3'_'$4'_'$5'_'$6'_'$7'_'$8'_'$9'_'$DATASET
MODEL_DIR='../results/models/'$DATASET'/models_round2_5/'$model_name'/'
LOG=$MODEL_DIR'/log_'$RAND
mkdir $MODEL_DIR
python train_model.py --model_dir=$MODEL_DIR --name=$NAME  --lr=$LR --epochs=$EPOCHS --model_type=$MODEL --probe_hidden_layers=$LAYERS --probe_hidden_multiplier=$MULT --supervision=$SUPERVISION --pretrained=$PRETRAINED --noise_std=$NOISE_STD --dataset=$DATASET| tee $LOG
