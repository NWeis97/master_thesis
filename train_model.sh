#!/bin/bash
export R=19
### NAME OF FILE
#BSUB -J config19
### OUTPUT AND ERROR FILE
#BSUB -o config_hpc/config19.out
#BSUB -e config_hpc/config19.err
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

rm config_hpc/config${R}.out
rm config_hpc/config${R}.err
source init.sh
wandb online

python3 src/training_test/train_classifier_model2.py --config-filename=training${R}