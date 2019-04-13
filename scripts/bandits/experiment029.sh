cd .. 
mkdir -p results/Experiment029
python3 bandits_experiment.py \
     --experiment_name Experiment029 \
     --optimizer_type  Adam \
     --eg_learning_rate 1e-3 \
     --eg_epsilon 1e-3 \
     --bnn_learning_rate 2e-4 \
     --bnn_epsilon 1e-4 \
     --bnn_lr_scheduler_step_size 10000 \
     --bnn_pi 0.5 \
     --bnn_log_sigma1 -3 \
     --bnn_log_sigma2 -8 \
     --averaged_weights \
     --initial_rho_weights_range -7 -6\
     --initial_mu_weights_range -0.05 0.05\
     --initial_rho_bias_range -7 -6\
     --initial_mu_bias_range -0.05 0.05 \
     --number_of_runs 4 \
> results/Experiment029/logs.txt 