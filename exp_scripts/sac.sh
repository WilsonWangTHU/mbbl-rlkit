for seed in 1234 2341 3412 4123; do
    for env_type in $1; do
        python examples/sac_bm.py --env_name $env_type --seed $seed --num_epochs $2
    done
done
