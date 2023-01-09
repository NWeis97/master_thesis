#!/bin/bash
export R=_BTM_iso_tune7
### NAME OF FILE
#BSUB -J config_BTM_iso_tune7
### OUTPUT AND ERROR FILE
#BSUB -o config_hpc/config_BTM_iso_tune7.out
#BSUB -e config_hpc/config_BTM_iso_tune7.err
### QUEUE TO BE USED
#BSUB -q gpuv100
### gpu memory
#BSUB -R "select[gpu32gb]"
### WALL TIME  REQUEST
#BSUB -W 24:00
### MEMORY REQUEST
#BSUB -R "rusage[mem=6000MB]"
### NUMBER OF CORES
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### SEND EMAIL
#BSUB -u s174466@student.dtu.dk
### SEND NOTIFICATION UPON COMPLETION
#BSUB -N

rm config_hpc/config${R}.out
rm config_hpc/config${R}.err
source init.sh
wandb online

# Run model
out=( $(python3 src/training_test/train_classifier_model_tune.py --config-filename=training${R}) )
model_name=${out[0]}

# Test model
export model_database=TRAIN
export balanced_dataset=-850
export test_dataset=val
export num_NN=425
export num_NN_kNN=4250
export num_MC=3000
export dist_classes=nn


# Test model
python3 src/training_test/test_classifier_model_tune.py --model-name=${model_name} --model-database=${model_database} --balanced-dataset=${balanced_dataset} --test-dataset=${test_dataset} --num-NN=${num_NN} --num-MC=${num_MC} --method='mixed' --with_OOD='False' --dist_classes=${dist_classes} --calibration_method='None'
python3 src/training_test/test_classifier_model_tune.py --model-name=${model_name} --model-database=${model_database} --balanced-dataset=${balanced_dataset} --test-dataset=${test_dataset} --num-NN=${num_NN} --num-MC=${num_MC} --method='min_dist_NN' --with_OOD='False' --dist_classes=${dist_classes} --calibration_method='None'
python3 src/training_test/test_classifier_model_tune.py --model-name=${model_name} --model-database=${model_database} --balanced-dataset=${balanced_dataset} --test-dataset=${test_dataset} --num-NN=${num_NN_kNN} --num-MC=${num_MC} --method='kNN_gauss_kernel' --with_OOD='False' --dist_classes=${dist_classes} --calibration_method='None'
