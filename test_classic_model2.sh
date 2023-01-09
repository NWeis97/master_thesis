#!/bin/bash
### Settings for test run
export R=1
export model_name=helpful-moon-1980
export model_database=TRAINVAL
export test_dataset=test
export calibration_method=TempScaling

#None
#TempScaling
#MCDropout
#SWAG


### NAME OF FILE
#BSUB -J config_test_classic1
### OUTPUT AND ERROR FILE
#BSUB -o config_hpc/config_test_classic1.out
#BSUB -e config_hpc/config_test_classic1.err
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

rm config_hpc/config_test_classic${R}.out
rm config_hpc/config_test_classic${R}.err
source init.sh

python3 src/training_test/test_classic_classifier_model.py --model-name="atomic-water-2218" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method=${calibration_method}
python3 src/training_test/test_classic_classifier_model.py --model-name="autumn-breeze-2223" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method=${calibration_method}
python3 src/training_test/test_classic_classifier_model.py --model-name="bumbling-cherry-2088" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method=${calibration_method}
python3 src/training_test/test_classic_classifier_model.py --model-name="bumbling-pyramid-2220" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method=${calibration_method}
python3 src/training_test/test_classic_classifier_model.py --model-name="curious-water-2098" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method=${calibration_method}
python3 src/training_test/test_classic_classifier_model.py --model-name="eternal-surf-2222" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method=${calibration_method}
python3 src/training_test/test_classic_classifier_model.py --model-name="helpful-moon-1980" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method=${calibration_method}
python3 src/training_test/test_classic_classifier_model.py --model-name="radiant-pond-2221" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method=${calibration_method}
python3 src/training_test/test_classic_classifier_model.py --model-name="toasty-sound-2219" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method=${calibration_method}
python3 src/training_test/test_classic_classifier_model.py --model-name="worldly-glitter-2103" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method=${calibration_method}

python3 src/training_test/test_classic_classifier_model.py --model-name="atomic-water-2218" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="None"
python3 src/training_test/test_classic_classifier_model.py --model-name="autumn-breeze-2223" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="None"
python3 src/training_test/test_classic_classifier_model.py --model-name="bumbling-cherry-2088" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="None"
python3 src/training_test/test_classic_classifier_model.py --model-name="bumbling-pyramid-2220" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="None"
python3 src/training_test/test_classic_classifier_model.py --model-name="curious-water-2098" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="None"
python3 src/training_test/test_classic_classifier_model.py --model-name="eternal-surf-2222" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="None"
python3 src/training_test/test_classic_classifier_model.py --model-name="helpful-moon-1980" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="None"
python3 src/training_test/test_classic_classifier_model.py --model-name="radiant-pond-2221" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="None"
python3 src/training_test/test_classic_classifier_model.py --model-name="toasty-sound-2219" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="None"
python3 src/training_test/test_classic_classifier_model.py --model-name="worldly-glitter-2103" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="None"

python3 src/training_test/test_classic_classifier_model.py --model-name="atomic-water-2218" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="SWAG"
python3 src/training_test/test_classic_classifier_model.py --model-name="autumn-breeze-2223" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="SWAG"
python3 src/training_test/test_classic_classifier_model.py --model-name="bumbling-cherry-2088" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="SWAG"
python3 src/training_test/test_classic_classifier_model.py --model-name="bumbling-pyramid-2220" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="SWAG"
python3 src/training_test/test_classic_classifier_model.py --model-name="curious-water-2098" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="SWAG"
python3 src/training_test/test_classic_classifier_model.py --model-name="eternal-surf-2222" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="SWAG"
python3 src/training_test/test_classic_classifier_model.py --model-name="helpful-moon-1980" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="SWAG"
python3 src/training_test/test_classic_classifier_model.py --model-name="radiant-pond-2221" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="SWAG"
python3 src/training_test/test_classic_classifier_model.py --model-name="toasty-sound-2219" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="SWAG"
python3 src/training_test/test_classic_classifier_model.py --model-name="worldly-glitter-2103" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="SWAG"

python3 src/training_test/test_classic_classifier_model.py --model-name="atomic-water-2218" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="MCDropout"
python3 src/training_test/test_classic_classifier_model.py --model-name="autumn-breeze-2223" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="MCDropout"
python3 src/training_test/test_classic_classifier_model.py --model-name="bumbling-cherry-2088" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="MCDropout"
python3 src/training_test/test_classic_classifier_model.py --model-name="bumbling-pyramid-2220" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="MCDropout"
python3 src/training_test/test_classic_classifier_model.py --model-name="curious-water-2098" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="MCDropout"
python3 src/training_test/test_classic_classifier_model.py --model-name="eternal-surf-2222" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="MCDropout"
python3 src/training_test/test_classic_classifier_model.py --model-name="helpful-moon-1980" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="MCDropout"
python3 src/training_test/test_classic_classifier_model.py --model-name="radiant-pond-2221" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="MCDropout"
python3 src/training_test/test_classic_classifier_model.py --model-name="toasty-sound-2219" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="MCDropout"
python3 src/training_test/test_classic_classifier_model.py --model-name="worldly-glitter-2103" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="MCDropout"

python3 src/training_test/test_classic_classifier_model.py --model-name="helpful-moon-1980" --model-database=${model_database} --test-dataset=${test_dataset} --calibration_method="Ensemble"