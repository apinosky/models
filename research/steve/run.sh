#!/bin/sh

# hearbeat to wait for some process to finish before continuing
# pid=15254
# while ps -p $pid; do sleep 60; done

# main loop
for env in 'halfcheetah' 'hopper' #
do
    for i in $(seq 13 100 50)
    do
        for method in   'ddpg'  #'steve' 'mve_tdk' #
        do
            echo $env $i $method
            # python3 master.py config/experiments/goodruns/$env/$method.json 0 $i --orig --overwrite_root "devel/H3/output_master_orig"
            python3 train.py config/new_experiments/$env\_h3/$method.json 0 $i --overwrite_root "devel/H3/output_train_m22p12_modPretrain1k_update250_modLR2" --no_done
            # python3 train.py config/new_experiments/$env/$method.json 0 $i --no_done # --render
        done
    done
done
