for env_type in gym_petsReacher gym_petsCheetah gym_petsPusher; do
    for seed in 1234 2341 3412 4123; do
        python examples/sac_bm.py --env_name $env_type --seed $seed
    done
done
