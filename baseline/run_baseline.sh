#!/bin/bash

# Use the baseline folder as context root
BASELINE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Remove the possible 'cp -i' alias
unalias cp 2>/dev/null

# Install base dependencies
pip install -r $BASELINE_ROOT/requirement.txt

# Enter data directory
pushd data

# Download public data in development phase
wget -q https://codalab.lisn.upsaclay.fr/my/datasets/download/eea9f5b7-3933-47cf-ba6f-394218eeb913 -O public_data_dev.zip
unzip public_data_dev.zip

# Preprocess offline data
python $BASELINE_ROOT/data_preprocess.py offline_592_1000.csv

# Download Revive SDK
wget https://www.revive.cn/statics/revive-sdk-0.5.0.zip
unzip revive-sdk-0.5.0.zip -x '__MACOSX/**'
find ./revive-sdk-0.5.0 -name 'revive*.zip' -exec unzip {} -d $BASELINE_ROOT \;
find ./revive-sdk-0.5.0 -name 'License*.zip' -exec unzip {} \;
rm -r revive-sdk-0.5.0

# Install Revive SDK and its dependencies
pushd $BASELINE_ROOT/revive
pip install -e .
export PYARMOR_LICENSE=$BASELINE_ROOT/data/license.lic
popd

# Start learning virtual environment (use ctrl+z to bring it to background)
python $BASELINE_ROOT/revive/train.py --data_file venv.npz --config_file venv.yaml --run_id venv_baseline --venv_mode tune --policy_mode None

# Acquire learned virtual environment
cp -f $BASELINE_ROOT/revive/logs/venv_baseline/env.pkl venv.pkl

# Go back to baseline directory
popd

# Start learning policy validation based on virtual environment (use ctrl+z to bring it to background)
mkdir -p data/logs
mkdir -p data/model_checkpoints
python train_policy.py

# Acquire learned policy validation
pushd $BASELINE_ROOT/data/model_checkpoints
cp -f $(ls -Art . | tail -n 1) $BASELINE_ROOT/data/rl_model.zip
popd

# Update sample submission and create bundle
pushd $BASELINE_ROOT/../sample_submission
cp -f $BASELINE_ROOT/data/evaluation_start_states.npy ./data/evaluation_start_states.npy
cp -f $BASELINE_ROOT/data/rl_model.zip ./data/rl_model.zip
zip -o -r --exclude='*.git*' --exclude='*__pycache__*' --exclude='*.DS_Store*' --exclude='*public_data*' ../sample_submission .;
popd
