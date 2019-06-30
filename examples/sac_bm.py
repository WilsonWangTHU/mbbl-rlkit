
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


def get_env(env_name, seed):

    if env_name in ['gym_walker2d', 'gym_hopper', 'gym_swimmer',
                    'gym_cheetah', 'gym_ant']:
        from mbbl.env.gym_env.walker import env
    elif env_name in ['gym_reacher']:
        from mbbl.env.gym_env.reacher import env
    elif env_name in ['gym_acrobot']:
        from mbbl.env.gym_env.acrobot import env
    elif env_name in ['gym_cartpole']:
        from mbbl.env.gym_env.cartpole import env
    elif env_name in ['gym_mountain']:
        from mbbl.env.gym_env.mountain_car import env
    elif env_name in ['gym_pendulum']:
        from mbbl.env.gym_env.pendulum import env
    elif env_name in ['gym_invertedPendulum']:
        from mbbl.env.gym_env.invertedPendulum import env
    elif env_name in ['gym_fwalker2d', 'gym_fhopper', 'gym_fant']:
        from mbbl.env.gym_env.fixed_walker import env
    elif env_name in ['gym_fswimmer']:
        from mbbl.env.gym_env.fixed_swimmer import env
    elif env_name in ['gym_humanoid', 'gym_slimhumanoid', 'gym_nostopslimhumanoid']:
        from mbbl.env.gym_env.humanoid import env
    else:
        from mbbl.env.gym_env.pets import env

    env = env(env_name=env_name, rand_seed=seed, misc_info={'reset_type': 'gym'})
    return env


def experiment(variant):

    expl_env = NormalizedBoxEnv(get_env(variant['env_name'], variant['seed']))
    eval_env = NormalizedBoxEnv(get_env(variant['env_name'], variant['seed']))

    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='test.')
    parser.add_argument('--env_name', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num_epochs', type=int, default=201)
    args = parser.parse_args()

    # noinspection PyTypeChecker
    variant = dict(
        algorithm="SAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=args.num_epochs,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    variant['env_name'] = args.env_name
    variant['seed'] = args.seed
    setup_logger('sac_' + args.env_name + "_seed_" + str(args.seed),
                 variant=variant)
    experiment(variant)
