cd ../..
mkdir -p results/bandits/Experiment006
python3 src/bandits/bandits_experiment.py \
     --experiment_name Experiment006 \
     --optimizer_type  Adam \
     --eg_learning_rate 1e-3 \
     --eg_epsilon 1e-3 \
     --bnn_learning_rate 1e-3 \
     --bnn_epsilon 1e-3 \
     --bnn_lr_scheduler_step_size 32 \
     --bnn_pi 0.75 \
     --bnn_log_sigma1 -2 \
     --bnn_log_sigma2 -7 \
     --averaged_weights \
     --initial_rho_weights_range -10 -2 \
     --initial_rho_weights_range -10 -2 \
> results/bandits/Experiment006/logs.txt 