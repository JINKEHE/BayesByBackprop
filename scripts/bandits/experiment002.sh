cd ../..
mkdir -p results/bandits/Experiment002
python3 src/bandits/bandits_experiment.py \
     --experiment_name Experiment002 \
     --optimizer_type  Adam \
     --eg_learning_rate 1e-3 \
     --eg_epsilon 1e-3 \
     --bnn_learning_rate 1e-3 \
     --bnn_epsilon 1e-3 \
     --bnn_lr_scheduler_step_size 32 \
     --bnn_pi 0.75 \
     --bnn_log_sigma1 -3 \
     --bnn_log_sigma2 -8 \
> results/bandits/Experiment002/logs.txt 
