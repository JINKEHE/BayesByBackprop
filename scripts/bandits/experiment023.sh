cd .. 
mkdir -p results/Experiment023
python3 bandits_experiment.py \
     --experiment_name Experiment023 \
     --optimizer_type  SGD \
     --eg_learning_rate 1e-3 \
     --bnn_learning_rate 2e-7 \
     --bnn_lr_scheduler_step_size 10000 \
     --bnn_pi 0.5 \
     --bnn_log_sigma1 -3 \
     --bnn_log_sigma2 -8 \
     --averaged_weights \
     --initial_rho_weights_range -7 -6\
     --initial_mu_weights_range -0.05 0.05\
     --initial_rho_bias_range -7 -6\
     --initial_mu_bias_range -0.05 0.05 \
     --number_of_runs 2 \
> results/Experiment023/logs.txt 