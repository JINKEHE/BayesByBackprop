cd ../src/mnist/
python mnist_experiment.py \
    --num_epochs 600 \
    --experiment_name "lr000025" \
    --num_units 1200 \
    --lr 0.00025 \
    --batch_size 128 \
    --network_type bayesian \
    --optimizer SGD \
    --num_samples_training 1 \
    --num_samples_testing 100 \
    --prior_type scale_mixture \
    --scale_mixture_pi 0.75 \
    --scale_mixture_log_sigma1 -0.0 \
    --scale_mixture_log_sigma2 -8.0 \
    --initial_rho_weights -3.5 -3 \
    --initial_rho_bias -3.5 -3 \
    --preprocess \
