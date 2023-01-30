#!/bin/bash
export R=_BTM_iso_train
### NAME OF FILE
#BSUB -J config_BTM_iso_train
### OUTPUT AND ERROR FILE
#BSUB -o config_hpc/config_BTM_iso_train.out
#BSUB -e config_hpc/config_BTM_iso_train.err
### QUEUE TO BE USED
#BSUB -q gpuv100
### gpu memory
#BSUB -R "select[gpu32gb]"
### WALL TIME  REQUEST
#BSUB -W 24:00
### MEMORY REQUEST
#BSUB -R "rusage[mem=3000MB]"
### NUMBER OF CORES
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### SEND EMAIL
#BSUB -u s174466@student.dtu.dk
### SEND NOTIFICATION UPON COMPLETION
#BSUB -N

source init.sh
if [ -e config_hpc/config${R}.out ]
then
    echo "Removing old log files...\n\n"
    rm config_hpc/config${R}.out
    rm config_hpc/config${R}.err
fi
wandb online

python3 src/training_test/train_classifier_model.py --config-filename=training${R}

