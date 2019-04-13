cd ../..
mkdir -p results/bandits/Experiment008
python3 src/bandits/bandits_experiment.py \
     --experiment_name Experiment008 \
     --optimizer_type  Adam \
     --eg_learning_rate 1e-3 \
     --eg_epsilon 1e-3 \
     --bnn_learning_rate 1e-3 \
     --bnn_epsilon 1e-3 \
     --bnn_lr_scheduler_step_size 32 \
     --bnn_pi 0.5 \
     --bnn_log_sigma1 -1 \
     --bnn_log_sigma2 -8 \
     --averaged_weights \
     --initial_rho_weights_range -8 -6\
     --initial_rho_bias_range -8 -6\
> results/bandits/Experiment008/logs.txt 