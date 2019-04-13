cd ../src/mnist/
python mnist_experiment.py \
    --num_epochs 600 \
    --num_units 1200 \
    --lr 1e-3 \
    --batch_size 128 \
    --network_type standard \
    --optimizer SGD \
    --experiment_name "SGD_baseline" \
    --save_weights \
    --preprocess \
