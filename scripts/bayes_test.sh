cd ../src/mnist/
python mnist_experiment.py \
    --num_epochs 600 \
    --num_units 1200 \
    --lr 0.0001 \
    --batch_size 128 \
    --network_type bayesian \
    --optimizer SGD \
    --num_samples_training 2 \
    --num_samples_testing 10 \
    --prior_type gaussian \
    --gaussian_mean 0.0 \
    --gaussian_log_sigma 0.0 \
