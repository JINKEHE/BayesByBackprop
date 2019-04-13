cd ../..
mkdir -p results/bandits/Experiment018
python3 src/bandits/bandits_experiment.py \
     --experiment_name Experiment018 \
     --optimizer_type  SGD \
     --eg_learning_rate 1e-3 \
     --bnn_learning_rate 1e-7 \
     --bnn_lr_scheduler_step_size 10000 \
     --bnn_pi 0.5 \
     --bnn_log_sigma1 -3 \
     --bnn_log_sigma2 -8 \
     --alpha 0.2 \
     --averaged_weights \
     --initial_rho_weights_range -10 -9\
     --initial_rho_bias_range -10 -9\
     --number_of_runs 4 \
> results/bandits/Experiment018/logs.txt 