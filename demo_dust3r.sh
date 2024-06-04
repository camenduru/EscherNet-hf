#!/bin/bash
# bash script to evaluate the model

# TODO ################### Chose CaPE type  ##########################

# 6DoF
cape_type="6DoF"
pretrained_model="XY-Xin/N3M3B112G6_6dof_36k"

# dust3r
data_type="dust3r"
T_ins=(1)
data_dir="/home/xin/code/dreammapping/dust3r_mine/dust3r/logs_dust3r/"  # TODO: change this to the path of the dust3r logs


# run
for T_in in "${T_ins[@]}"; do
    python eval_eschernet.py --pretrained_model_name_or_path "$pretrained_model" \
                         --data_dir "$data_dir" \
                         --data_type "$data_type" \
                          --cape_type "$cape_type" \
                         --T_in "$T_in"
done