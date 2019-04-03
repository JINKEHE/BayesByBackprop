cd ../src/mnist/
python mnist_experiment.py \
    --num_epochs 600 \
    --num_units 1200 \
    --lr 0.0025 \
    --batch_size 128 \
    --network_type bayesian \
    --optimizer SGD \
    --num_samples_training 4 \
    --num_samples_testing 4 \
    --prior_type scale_mixture \
    --scale_mixture_pi 0.5 \
    --scale_mixture_log_sigma1 -2.0 \
    --scale_mixture_log_sigma2 -8.0 \
    --preprocess \
