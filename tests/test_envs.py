from stable_baselines3.common.env_checker import check_env

from blackjack.envs import ENVS


def test_all_registered_envs_pass_check_env():
    for make in ENVS.values():
        check_env(make())
