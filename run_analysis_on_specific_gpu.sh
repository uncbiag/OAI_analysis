#! /usr/bin/env bash

# we assume we have 4 GPUs here, so interval is 4, offset will be the GPU number

source ~/.bashrc
source activate oai_mn

CUDA_VISIBLE_DEVICES=$1 python mn_oai_pipeline.py --only_recompute_if_thickness_file_is_missing --knee_type LEFT_KNEE --progression_cohort_only --data_division_interval 4 --data_division_offset $1
