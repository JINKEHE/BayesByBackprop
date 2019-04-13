cd .. 
mkdir -p results/Experiment005
python3 bandits_experiment.py \
     --experiment_name Experiment005 \
     --optimizer_type  Adam \
     --eg_learning_rate 1e-3 \
     --eg_epsilon 1e-3 \
     --bnn_learning_rate 1e-3 \
     --bnn_epsilon 1e-3 \
     --bnn_lr_scheduler_step_size 32 \
     --bnn_pi 0.75 \
     --bnn_log_sigma1 -3 \
     --bnn_log_sigma2 -8 \
     --averaged_weights \
     --initial_rho_weights_range -7 -4 \
     --initial_rho_weights_range -7 -4 \
> results/Experiment005/logs.txt 