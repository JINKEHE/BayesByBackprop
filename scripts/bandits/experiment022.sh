cd .. 
mkdir -p results/Experiment022
python3 bandits_experiment.py \
     --experiment_name Experiment022 \
     --optimizer_type  SGD \
     --eg_learning_rate 1e-3 \
     --bnn_learning_rate 2e-7 \
     --bnn_lr_scheduler_step_size 10000 \
     --bnn_pi 0.5 \
     --bnn_log_sigma1 -3 \
     --bnn_log_sigma2 -8 \
     --averaged_weights \
     --initial_rho_weights_range -10 -9\
     --initial_rho_bias_range -10 -9\
     --number_of_runs 1 \
> results/Experiment022/logs.txt 