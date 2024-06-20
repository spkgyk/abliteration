#!/bin/sh

ENV_NAME="abliteration"

conda activate base
conda remove -n $ENV_NAME --all --yes
conda env create -f setup/requirements.yaml
conda activate $ENV_NAME