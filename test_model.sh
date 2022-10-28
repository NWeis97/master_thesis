#!/bin/bash
### Settings for test run
export R=11
export model_name=ruby-forest-460
export model_database=TRAINVAL
export balanced_dataset=0
export test_dataset=test
export num_NN=300
export num_MC=1000
export method=min_dist_NN
export with_OOD=False
###min_dist_NN_pr_class
###min_dist_NN_cap_class_NN
###min_dist_NN_with_min_dist_rand
###avg_dist_rand
###min_dist_NN

### NAME OF FILE
#BSUB -J config_test1
### OUTPUT AND ERROR FILE
#BSUB -o config_hpc/config_test1.out
#BSUB -e config_hpc/config_test1.err
### QUEUE TO BE USED
#BSUB -q gpuv100
### gpu memory
#BSUB -R "select[gpu32gb]"
### WALL TIME  REQUEST
#BSUB -W 3:00
### MEMORY REQUEST
#BSUB -R "rusage[mem=4000MB]"
### NUMBER OF CORES
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1"
### SEND EMAIL
#BSUB -u s174466@student.dtu.dk
### SEND NOTIFICATION UPON COMPLETION
#BSUB -N

rm config_hpc/config_test${R}.out
rm config_hpc/config_test${R}.err
source init.sh

python3 src/training_test/test_classifier_model.py --model-name=${model_name} --model-database=${model_database} --balanced-dataset=${balanced_dataset} --test-dataset=${test_dataset} --num-NN=${num_NN} --num-MC=${num_MC} --method=${method} --with_OOD=${with_OOD}