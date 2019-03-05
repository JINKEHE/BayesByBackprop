cd ../src/mnist/
python mnist_experiment.py \
    --num_epochs 11 \
    --num_units 1200 \
    --lr 1e-3 \
    --batch_size 128 \
    --network_type bayesian \
    --optimizer SGD \
