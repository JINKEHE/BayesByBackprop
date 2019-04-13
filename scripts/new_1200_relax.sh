cd ../src/mnist/
python mnist_experiment.py \
    --num_epochs 600 \
    --num_units 1200 \
    --lr 0.001 \
    --batch_size 128 \
    --network_type bayesian \
    --optimizer SGD \
    --num_samples_training 10 \
    --num_samples_testing 10 \
    --prior_type scale_mixture \
    --scale_mixture_pi 0.75 \
    --scale_mixture_log_sigma1 -2.0 \
    --scale_mixture_log_sigma2 -7.0 \
    --preprocess \
