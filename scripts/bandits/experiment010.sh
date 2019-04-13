cd .. 
mkdir -p results/Experiment010
python3 bandits_experiment.py \
     --experiment_name Experiment010 \
     --optimizer_type  SGD \
     --eg_learning_rate 1e-3 \
     --eg_momentum 0 \
     --bnn_learning_rate 1e-6 \
     --bnn_lr_scheduler_step_size 128 \
     --bnn_lr_scheduler_gamma 0.9 \
     --bnn_momentum 0 \
     --bnn_pi 0.5 \
     --bnn_log_sigma1 -3 \
     --bnn_log_sigma2 -8 \
     --averaged_weights \
     --initial_rho_weights_range -10 -9\
     --initial_rho_bias_range -10 -9\
> results/bandits/Experiment010/logs.txt 