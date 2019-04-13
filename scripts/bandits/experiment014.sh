cd ../..
mkdir -p results/bandits/Experiment014
python3 src/bandits/bandits_experiment.py \
     --experiment_name Experiment014 \
     --optimizer_type  SGD \
     --eg_learning_rate 1e-3 \
     --eg_momentum 0.9 \
     --bnn_learning_rate 1e-7 \
     --bnn_lr_scheduler_step_size 24 \
     --bnn_lr_scheduler_gamma 0.9 \
     --bnn_momentum 0.9 \
     --bnn_pi 0.5 \
     --bnn_log_sigma1 -3 \
     --bnn_log_sigma2 -8 \
     --averaged_weights \
     --initial_rho_weights_range -10 -9\
     --initial_rho_bias_range -10 -9\
     --number_of_runs 4 \
> results/bandits/Experiment014/logs.txt 