#!/bin/bash
### Settings for test run
export R=1
export model_name=copper-vortex-1980
export model_database=TRAINVAL
export balanced_dataset=-1350
export test_dataset=test
export num_NN=675
export num_MC=3000
export num_NN_kNN=6750
export method=min_dist_NN
export with_OOD=False
export dist_classes=nn
export calibration_method=Ensemble

#kNN_gauss_kernel
#min_dist_NN

#SWAG
#None
#MCDropout

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
#BSUB -R "rusage[mem=20000MB]"
### NUMBER OF CORES
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### SEND EMAIL
#BSUB -u s174466@student.dtu.dk
### SEND NOTIFICATION UPON COMPLETION
#BSUB -N

rm config_hpc/config_test${R}.out
rm config_hpc/config_test${R}.err
source init.sh

python3 src/training_test/test_classifier_model.py --model-name=${model_name} --model-database=${model_database} --balanced-dataset=${balanced_dataset} --test-dataset=${test_dataset} --num-NN=${num_NN} --num-MC=${num_MC} --method=${method} --with_OOD=False --dist_classes=${dist_classes} --calibration_method=${calibration_method}
python3 src/training_test/test_classifier_model.py --model-name=${model_name} --model-database=${model_database} --balanced-dataset=${balanced_dataset} --test-dataset=${test_dataset} --num-NN=${num_NN} --num-MC=${num_MC} --method=${method} --with_OOD=True --dist_classes=${dist_classes} --calibration_method=${calibration_method}
python3 src/training_test/test_classifier_model.py --model-name=${model_name} --model-database=${model_database} --balanced-dataset=${balanced_dataset} --test-dataset=${test_dataset} --num-NN=${num_NN_kNN} --num-MC=${num_MC} --method=kNN_gauss_kernel --with_OOD=False --dist_classes=${dist_classes} --calibration_method=${calibration_method}
python3 src/training_test/test_classifier_model.py --model-name=${model_name} --model-database=${model_database} --balanced-dataset=${balanced_dataset} --test-dataset=${test_dataset} --num-NN=${num_NN_kNN} --num-MC=${num_MC} --method=kNN_gauss_kernel --with_OOD=True --dist_classes=${dist_classes} --calibration_method=${calibration_method}


python3 src/training_test/test_classifier_model.py --model-name=warm-wind-1979 --model-database=${model_database} --balanced-dataset=${balanced_dataset} --test-dataset=${test_dataset} --num-NN=${num_NN} --num-MC=${num_MC} --method=${method} --with_OOD=False --dist_classes=${dist_classes} --calibration_method=${calibration_method}
python3 src/training_test/test_classifier_model.py --model-name=warm-wind-1979 --model-database=${model_database} --balanced-dataset=${balanced_dataset} --test-dataset=${test_dataset} --num-NN=${num_NN} --num-MC=${num_MC} --method=${method} --with_OOD=True --dist_classes=${dist_classes} --calibration_method=${calibration_method}
python3 src/training_test/test_classifier_model.py --model-name=warm-wind-1979 --model-database=${model_database} --balanced-dataset=${balanced_dataset} --test-dataset=${test_dataset} --num-NN=${num_NN_kNN} --num-MC=${num_MC} --method=kNN_gauss_kernel --with_OOD=False --dist_classes=${dist_classes} --calibration_method=${calibration_method}
python3 src/training_test/test_classifier_model.py --model-name=warm-wind-1979 --model-database=${model_database} --balanced-dataset=${balanced_dataset} --test-dataset=${test_dataset} --num-NN=${num_NN_kNN} --num-MC=${num_MC} --method=kNN_gauss_kernel --with_OOD=True --dist_classes=${dist_classes} --calibration_method=${calibration_method}