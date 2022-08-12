#/bin/bash
DATASET='dsprites'
NAME='round2_5'
NOISE_STD=0.001
SUPERVISION='unsup'
EPOCHS=15
LR=0.001
PRETRAINED=no


for MODEL in raw_data noisy_labels noisy_uniform_mix_labels random_linear_mix_labels
do
	if [ $MODEL == 'noisy_labels' ]; 
	then
		NOISE_STD=0.2
	else
		NOISE_STD=0.003
	fi

	# separate probing of linear layers -> hidden dim == 0
	# for linear layers MULT doesn't do anything
	echo $NAME $MODEL 0 1 $SUPERVISION $EPOCHS $LR $PRETRAINED $NOISE_STD $DATASET
	bash start_exp.sh $NAME $MODEL 0 1 $SUPERVISION $EPOCHS $LR $PRETRAINED $NOISE_STD $DATASET
	# for LAYERS in {1..4}
	for LAYERS in 2
	do
		for MULT in 2 4 8 12 16 32 64 128 256 512

		do
			echo $NAME $MODEL $LAYERS $MULT $SUPERVISION $EPOCHS $LR $PRETRAINED $NOISE_STD $DATASET
			bash start_exp.sh $NAME $MODEL $LAYERS $MULT $SUPERVISION $EPOCHS $LR $PRETRAINED $NOISE_STD $DATASET
		done

	done 
done


MODEL=conv_net
for SUPERVISION in unsup supervised
do
	# separate probing of linear layers -> hidden dim == 0
	# for linear layers MULT doesn't do anything
	echo $NAME $MODEL 0 1 $SUPERVISION $EPOCHS $LR $PRETRAINED $NOISE_STD $DATASET
	bash start_exp.sh $NAME $MODEL 0 1 $SUPERVISION $EPOCHS $LR $PRETRAINED $NOISE_STD $DATASET
	for LAYERS in 2
	do
		for MULT in 8 16 32 64 128 256 512
		do
			echo $NAME $MODEL $LAYERS $MULT $SUPERVISION $EPOCHS $LR $PRETRAINED $NOISE_STD $DATASET
			bash start_exp.sh $NAME $MODEL $LAYERS $MULT $SUPERVISION $EPOCHS $LR $PRETRAINED $NOISE_STD $DATASET
		done

	done 
done


MODEL=resnet18
for SUPERVISION in unsup supervised
do
	for PRETRAINED in yes no
	do
		# separate probing of linear layers -> hidden dim == 0
		# for linear layers MULT doesn't do anything
		echo $NAME $MODEL 0 1 $SUPERVISION $EPOCHS $LR $PRETRAINED $NOISE_STD $DATASET
		bash start_exp.sh $NAME $MODEL 0 1 $SUPERVISION $EPOCHS $LR $PRETRAINED $NOISE_STD $DATASET
		for LAYERS in 2
		do
			for MULT in 8 16 32 64 128 256 512
			do
				echo $NAME $MODEL $LAYERS $MULT $SUPERVISION $EPOCHS $LR $PRETRAINED $NOISE_STD $DATASET
				bash start_exp.sh $NAME $MODEL $LAYERS $MULT $SUPERVISION $EPOCHS $LR $PRETRAINED $NOISE_STD $DATASET
			done

		done 
	done
done