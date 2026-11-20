# Blackjack RL — Assignment 2 (CC3042)

Customizar o `Blackjack-v1` do Gymnasium, treinar um agente com Stable-Baselines3 e comparar com o ambiente original. Enunciado: [sigarra/Assignment 2 Description.pdf](sigarra/Assignment%202%20Description.pdf).

## Setup

```bash
uv venv --python 3.14        # só se .venv não existir
uv pip install -r requirements.txt
```

Testado com Python 3.14, gymnasium 1.4.0, stable-baselines3 2.9.2, torch 2.15 (CPU).

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `blackjack/` | código: `envs.py` (`make_env` + wrappers de customização) |
| `docs/` | decisões e plano de desenvolvimento |
| `tests/` | testes (`.venv/bin/python -m pytest`) |
| `notebooks/` | experimentos e gráficos |
| `results/` | modelos, logs do tensorboard, CSVs (ignorado pelo git) |
| `sigarra/` | material da disciplina (enunciado, tutoriais) |

Nota: o SB3 não aceita a observação `Tuple` do Blackjack; `make_env` já aplica `FlattenObservation` (vetor one-hot de 45).

## Milestones

- [ ] **Semana 1 — Ambiente:** escolher e implementar customizações (reward, estado, regras)
- [ ] **Semana 2 — Agente:** escolher algoritmo (PPO / DQN / A2C), treinar, tunar hiperparâmetros
- [ ] **Semana 3 — Avaliação:** várias seeds, comparar original vs customizado, apresentação
