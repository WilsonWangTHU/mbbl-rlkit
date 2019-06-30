for env_type in gym_invertedPendulum gym_acrobot gym_cartpole gym_mountain; do
    for seed in 1234 2341 3412 4123; do
        python examples/td3_bm.py --env_name $env_type --seed $seed
    done
done
