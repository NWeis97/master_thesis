#!/bin/bash
### Settings for test run
export R=1
export model_name=golden-puddle-238
export model_database=TRAIN
export balanced_dataset=0
export test_dataset=test
export num_NN=30
export num_MC=1000

### NAME OF FILE
#BSUB -J config_test
### OUTPUT AND ERROR FILE
#BSUB -o config_hpc/config_test15.out
#BSUB -e config_hpc/config_test15.err
### QUEUE TO BE USED
#BSUB -q gpuv100
### gpu memory
#BSUB -R "select[gpu32gb]"
### WALL TIME  REQUEST
#BSUB -W 1:00
### MEMORY REQUEST
#BSUB -R "rusage[mem=3000MB]"
### NUMBER OF CORES
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### SEND EMAIL
#BSUB -u s174466@student.dtu.dk
### SEND NOTIFICATION UPON COMPLETION
#BSUB -N

rm config_hpc/config_test${R}.out
rm config_hpc/config_test${R}.err
source init.sh

python3 src/training_test/test_classifier_model.py --model-name=${model_name} --model-database=${model_database} --balanced-dataset=${balanced_dataset} --test-dataset=${test_dataset} --num-NN=${num_NN} --num-MC=${num_MC}