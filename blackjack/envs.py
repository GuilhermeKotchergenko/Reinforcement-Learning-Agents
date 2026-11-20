import gymnasium as gym
from gymnasium.wrappers import FlattenObservation


def make_env(**kwargs) -> gym.Env:
    """Blackjack-v1 original, pronto para o SB3.

    O SB3 não aceita observação Tuple (soma, carta do dealer, ás usável),
    então achatamos para um vetor one-hot. As customizações entram aqui
    como wrappers extras (reward, estado, regras).
    """
    return FlattenObservation(gym.make("Blackjack-v1", **kwargs))


# Cada customização registra aqui: nome -> função que cria o env já pronto para o SB3.
ENVS = {"original": make_env}
