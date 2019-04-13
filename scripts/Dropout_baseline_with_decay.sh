cd ../src/mnist/
python mnist_experiment.py \
    --num_epochs 600 \
    --num_units 1200 \
    --lr 1e-3 \
    --batch_size 128 \
    --network_type standard \
    --optimizer SGD \
    --use_dropout \
    --dropout_rate 0.5 \
    --experiment_name "Dropout_baseline_with_decay" \
    --lr_scheduler_step_size 300 \
    --save_weights \
    --preprocess \
