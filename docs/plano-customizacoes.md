# Plano de desenvolvimento: pipeline e customizações

Sete tarefas, cada uma para um agente. A **Tarefa 0** (pipeline de treino e avaliação) e as **Tarefas A–D** (ambientes) podem rodar em paralelo: os ambientes não dependem do pipeline para serem testados. A **Tarefa E** (desafios) mexe no `evaluate.py`, então só começa depois que a Tarefa 0 estiver na `main`. A **Tarefa F** (contar sem ser pego) reaproveita o sapato da Tarefa D, então só começa depois dela.

| Tarefa | Entrega | Tempo estimado | Depende de |
| --- | --- | --- | --- |
| 0. Pipeline | `train.py`, `evaluate.py`, `basic_strategy.py` | ~2h | — |
| A. Penalidade por carta | `blackjack/hit_penalty.py` | ~1h | — |
| B. Dealer H17 | `blackjack/dealer_h17.py` | ~2h | — |
| C. Double down | `blackjack/double_down.py` | ~3–4h | — |
| D. Baralho finito + contagem | `blackjack/card_counting.py` | ~1 dia | — |
| E. Desafios para o modelo treinado | `blackjack/challenges.py` | ~3h | Tarefa 0 |
| F. Contar cartas sem ser pego | `blackjack/casino.py` | ~2 dias | Tarefa D |

Para mandar um agente: *"Faça a Tarefa X de `docs/plano-customizacoes.md`. Leia antes a seção Convenções."*

---

## Convenções (valem para todas as tarefas)

**Ambiente de trabalho**
- Python 3.14 na `.venv`. Rode tudo com `.venv/bin/python`. Dependências em `requirements.txt` (instale com `uv pip install -r requirements.txt`).
- Cada agente trabalha no próprio branch: `feat/<nome-da-tarefa>` (ex.: `feat/hit-penalty`). Não faça merge na `main`.
- Testes: `.venv/bin/python -m pytest`. Todos precisam passar antes de entregar.

**Arquivos que cada tarefa pode tocar**
- O próprio módulo em `blackjack/` e o próprio teste em `tests/test_<nome>.py`.
- **Uma linha** em `ENVS` no fim de [blackjack/envs.py](../blackjack/envs.py), registrando o ambiente. Nada mais nesse arquivo.
- Qualquer outro arquivo: não mexa. Se precisar, pare e descreva a mudança no resumo final.

**Interface de cada ambiente customizado**
- Expor uma função `make_<nome>_env(**kwargs) -> gym.Env` que devolve o ambiente **já com `FlattenObservation`**, igual a `make_env` em [blackjack/envs.py](../blackjack/envs.py).
- Não importar `blackjack.envs` dentro do módulo (evita import circular). Importe direto de `gymnasium`.
- O ambiente precisa passar no `check_env` do SB3. O teste [tests/test_envs.py](../tests/test_envs.py) já faz isso para tudo que está em `ENVS`.
- **Resultado real da mão:** no passo final de cada episódio, colocar em `info["game_reward"]` o ganho/perda real do jogo (+1, 0, -1; ou ×2 se dobrou; +1,5 em natural se ativado). Isso é o que a avaliação mede. A recompensa de treino pode ser diferente (ex.: Tarefa A), mas o `game_reward` é sempre o dinheiro de verdade.
- Usar sempre `self.np_random` (ou `env.unwrapped.np_random`) para sortear cartas, para que `reset(seed=...)` torne o episódio reprodutível.

**Código**
- Comentários e docstrings em português; nomes de variáveis e funções em inglês.
- A lógica do jogo do Gymnasium está em `gymnasium/envs/toy_text/blackjack.py` (funções `draw_card`, `sum_hand`, `usable_ace`, `is_bust`, `score`, `cmp`, `is_natural`). Reaproveite essas funções em vez de reescrever.
- Hiperparâmetros dos algoritmos ficam nos padrões do SB3. Ajuste fino é outra etapa.

**Resumo final de cada agente**
1. Arquivos criados ou alterados.
2. Saída do `pytest`.
3. Qualquer decisão tomada que não estava no plano, com o motivo.

---

## Tarefa 0 — Pipeline de treino e avaliação

**Objetivo:** treinar PPO ou DQN em qualquer ambiente de `ENVS` e medir o resultado real de jogo, para comparar original vs customizado.

**Entregas**
1. `blackjack/train.py`, executável como:
   ```bash
   .venv/bin/python -m blackjack.train --env original --algo ppo --seed 0 --timesteps 200000
   ```
   - `--algo` aceita `ppo` e `dqn`.
   - Envolve o ambiente com `Monitor` do SB3 (log em `results/monitor/<env>_<algo>_s<seed>.monitor.csv`).
   - Log do TensorBoard em `results/tb/`.
   - Salva o modelo em `results/models/<env>_<algo>_s<seed>.zip`.
   - Seed aplicada ao algoritmo (`seed=`) e ao ambiente.
2. `blackjack/evaluate.py`, executável como:
   ```bash
   .venv/bin/python -m blackjack.evaluate --env original --algo ppo --seed 0 --episodes 100000
   ```
   - Carrega o modelo salvo e joga `--episodes` mãos com `deterministic=True`, numa seed de avaliação diferente da de treino.
   - Mede pelo `info["game_reward"]`; se a chave não existir, usa a recompensa do passo final.
   - Acrescenta uma linha em `results/eval.csv` com: `env, algo, seed, episodes, mean_return, std_return, win_rate, draw_rate, loss_rate`.
   - Aceita `--policy basic` no lugar de um modelo, para avaliar a basic strategy (item 3).
3. `blackjack/basic_strategy.py`: função `basic_strategy_action(player_sum, dealer_card, usable_ace) -> int` com a tabela de parar/pedir carta da basic strategy para baralho infinito e dealer que para em 17. Serve de referência no ambiente original (só ações 0 e 1).
4. `tests/test_pipeline.py`:
   - Treino de 1000 passos com PPO e com DQN no `original` roda sem erro e gera o `.zip`.
   - A avaliação da basic strategy em 50000 mãos dá `mean_return` entre -0,10 e 0,00 (valor conhecido: cerca de -0,05).
   - Use `tmp_path` do pytest para não sujar `results/`.

**Fora do escopo:** gráficos, notebooks, rodar todas as seeds, ajuste de hiperparâmetros.

**Pronto quando:** os dois comandos acima funcionam no `original` e o `pytest` passa.

---

## Tarefa A — Penalidade por pedir carta

**Hipótese:** com uma penalidade de -0,1 por carta pedida, o agente para mais cedo (fica mais conservador) e o retorno real de jogo piora um pouco em relação ao original.

**Entregas**
1. `blackjack/hit_penalty.py` com:
   - Um `gym.Wrapper` que soma `penalty` (padrão `-0.1`) à recompensa sempre que a ação for 1 (pedir carta).
   - Guarda o resultado real em `info["game_reward"]` no passo final (a recompensa original, sem a penalidade).
   - `make_hit_penalty_env(penalty=-0.1, **kwargs)`.
2. Registro: `"hit_penalty": make_hit_penalty_env` em `ENVS`.
3. `tests/test_hit_penalty.py`:
   - Pedir carta sem estourar dá recompensa -0,1.
   - No passo final, `game_reward` está em {-1, 0, 1}. Se a última ação foi pedir carta (estourou), `reward = game_reward - 0,1`; se foi parar, `reward = game_reward`.
   - `penalty=0` se comporta igual ao original com a mesma seed.

**Pronto quando:** o `pytest` passa e o ambiente aparece em `ENVS`.

---

## Tarefa B — Dealer pede carta com 17 "soft" (H17)

**Hipótese:** quando o dealer também pede carta com 17 "soft" (17 contando um ás como 11), o retorno médio do agente cai e a política muda nas mãos perto de 17.

**Atenção:** em cassinos essa regra muda a vantagem da casa em só ~0,2 ponto percentual. Com 100 mil mãos de avaliação o erro padrão é ~0,3 ponto, então a diferença pode ficar dentro do ruído. Registre isso no resumo final; a Tarefa 0 pode precisar de mais mãos para essa comparação.

**Entregas**
1. `blackjack/dealer_h17.py` com:
   - Uma subclasse de `BlackjackEnv` que sobrescreve `step`. Ao parar (ação 0), o dealer pede carta enquanto `sum_hand(dealer) < 17` **ou** (`sum_hand(dealer) == 17` **e** `usable_ace(dealer)`). O resto do `step` fica igual ao original.
   - `info["game_reward"]` no passo final (igual à recompensa, já que não há modelagem de recompensa).
   - `make_dealer_h17_env(**kwargs)`, com `FlattenObservation`. Não precisa registrar no `gym.register`; instancie a classe direto.
2. Registro: `"dealer_h17": make_dealer_h17_env` em `ENVS`.
3. `tests/test_dealer_h17.py`:
   - Montando a mão do dealer à mão (ex.: `[1, 6]`, 17 soft), ao parar o dealer pede mais carta.
   - Com `[10, 7]` (17 "hard"), o dealer para.
   - Com a mesma seed, uma mão em que o dealer nunca chega a 17 soft dá o mesmo resultado do original.

**Pronto quando:** o `pytest` passa e o ambiente aparece em `ENVS`.

---

## Tarefa C — Ação "double down"

**Hipótese:** com a opção de dobrar a aposta, o agente aprende a dobrar com 10 ou 11 contra cartas fracas do dealer (como a basic strategy) e o retorno médio melhora em relação ao original.

**Regras**
- Ações: `0` = parar, `1` = pedir carta, `2` = dobrar.
- Dobrar só é permitido na primeira decisão da mão (com 2 cartas). Recebe exatamente 1 carta, o dealer joga, e a recompensa é multiplicada por 2 (estourar ao dobrar = -2).
- Dobrar depois da primeira decisão é tratado como pedir carta (ação 1). O SB3 sem `sb3-contrib` não tem máscara de ações, então a ação inválida não pode quebrar o episódio.
- A observação ganha um campo `can_double` (`Discrete(2)`), para o agente saber se dobrar vale naquele momento. Fica: `(soma do jogador, carta do dealer, ás usável, can_double)`.

**Entregas**
1. `blackjack/double_down.py` com:
   - Uma subclasse de `BlackjackEnv` com `action_space = Discrete(3)` e o `observation_space` ampliado.
   - `info["game_reward"]` no passo final (igual à recompensa, já que o ×2 é dinheiro de verdade).
   - `make_double_down_env(**kwargs)`, com `FlattenObservation` (vetor de 47 posições).
2. Registro: `"double_down": make_double_down_env` em `ENVS`.
3. `tests/test_double_down.py`:
   - Dobrar na primeira decisão termina o episódio com recompensa em {-2, 0, 2}.
   - `can_double` é 1 no `reset` e 0 depois de pedir carta.
   - Ação 2 depois de pedir carta se comporta como ação 1.
   - Episódios só com ações 0 e 1 dão o mesmo resultado do original com a mesma seed.

**Aviso para a Tarefa 0:** a basic strategy da Tarefa 0 não usa dobrar. Para comparar com este ambiente, a referência justa é a basic strategy com double down; registre isso no resumo como pendência.

**Pronto quando:** o `pytest` passa e o ambiente aparece em `ENVS`.

---

## Tarefa D — Baralho finito com contagem de cartas

**Hipótese (revisada):** num baralho finito, o agente que vê a contagem de cartas obtém retorno médio maior que o mesmo agente sem a contagem.

**Por que revisada:** a ideia inicial era "superar a basic strategy". Sem poder variar a aposta, contar cartas só ajuda nas decisões de parar ou pedir carta, e o ganho esperado é pequeno (décimos de ponto percentual). A comparação justa é baralho finito **com** contagem vs baralho finito **sem** contagem. Variar a aposta conforme a contagem fica como extensão opcional, fora desta tarefa.

**Regras**
- Sapato de `n_decks` baralhos (padrão 6), embaralhado com `self.np_random`. As cartas saem do sapato sem reposição.
- Reembaralha no `reset` quando restar menos de 25% das cartas (penetração de 75%). O sapato **persiste entre episódios**; só é recriado ao reembaralhar.
- Contagem Hi-Lo: cartas 2–6 valem +1, 7–9 valem 0, 10 e ás valem -1. A contagem considera todas as cartas já vistas: as do jogador e a carta aberta do dealer durante a mão, e a mão inteira do dealer quando ela termina.
- "True count" = contagem corrida ÷ baralhos restantes, arredondada e limitada a [-5, 5]. Na observação vira `Discrete(11)` (valor + 5).

**Entregas**
1. `blackjack/card_counting.py` com:
   - Uma subclasse de `BlackjackEnv` que sobrescreve `reset` e `step` para tirar cartas do sapato em vez de `draw_card`.
   - Parâmetro `show_count: bool`. Com `True`, a observação é `(soma, carta do dealer, ás usável, true_count)`. Com `False`, a observação é a original, e o resto do jogo é idêntico (é o grupo de controle).
   - `info["game_reward"]` no passo final e `info["true_count"]` em todo passo (mesmo com `show_count=False`, para análise).
   - `make_card_counting_env(show_count=True, n_decks=6, **kwargs)`, com `FlattenObservation`.
   - O sapato e a contagem Hi-Lo ficam numa classe separada e reutilizável (ex.: `Shoe`, com `draw()`, `running_count`, `true_count` e `needs_shuffle`), porque a Tarefa F usa a mesma lógica.
2. Registro de **dois** ambientes em `ENVS`: `"finite_deck": lambda **kw: make_card_counting_env(show_count=False, **kw)` e `"card_counting": make_card_counting_env`.
3. `tests/test_card_counting.py`:
   - Um sapato de 1 baralho tem 52 cartas, com 4 de cada valor de 1 a 9 e 16 de valor 10.
   - Depois de várias mãos sem reembaralhar, o número de cartas restantes cai certinho.
   - A contagem corrida bate com o Hi-Lo calculado à mão sobre as cartas vistas.
   - Reembaralha quando passa da penetração, e a contagem volta a 0.
   - Mesma seed → mesma sequência de mãos.

**Pronto quando:** o `pytest` passa e os dois ambientes aparecem em `ENVS`.

---

## Tarefa E — Desafios para o modelo treinado

**Ideia:** as Tarefas A–D mudam o ambiente **antes** do treino. Aqui o agente é treinado normalmente e depois enfrenta um jogo alterado **sem retreinar**. A pergunta é: quanto ele perde quando o jogo muda sem aviso, e PPO ou DQN aguenta melhor?

Cada desafio tem uma intensidade `p` entre 0 e 1. Com `p = 0` o jogo é idêntico ao original.

| Desafio | O que acontece | Hipótese |
| --- | --- | --- |
| **Dealer trapaceiro** | Com probabilidade `p`, cada carta que o dealer tira é escolhida entre duas: fica com a melhor para ele (a que chega mais perto de 21 sem estourar) | O retorno cai de forma proporcional a `p`; o agente continua jogando como se o dealer fosse honesto |
| **Baralho viciado** | Os 10 (10, J, Q, K) ficam `p` × 50% menos prováveis | O retorno cai: menos 10 favorecem o dealer, e a política aprendida pede carta de menos |
| **Empate é derrota** | Com probabilidade `p`, um empate vira derrota (-1) | O retorno cai cerca de `p` × taxa de empate (~9%); a política não muda porque o agente não sabe da regra |

**Entregas**
1. `blackjack/challenges.py` com:
   - Um wrapper ou subclasse por desafio, cada um com parâmetro `p`.
   - Os desafios que mudam como as cartas saem (trapaceiro, baralho viciado) sobrescrevem o sorteio usando `self.np_random`.
   - `info["game_reward"]` no passo final, com o resultado já alterado pelo desafio.
   - Um dicionário `CHALLENGES = {"cheating_dealer": ..., "stacked_deck": ..., "push_loses": ...}`, cada valor uma função `make(p, **kwargs) -> gym.Env` já com `FlattenObservation`. **Não** registrar em `ENVS`: desafios não são ambientes de treino.
2. Em `blackjack/evaluate.py` (da Tarefa 0): opções `--challenge <nome>` e `--p <valor>`. O modelo treinado no ambiente `--env` é avaliado no ambiente do desafio. Acrescentar as colunas `challenge` e `p` em `results/eval.csv` (vazias quando não há desafio).
3. `tests/test_challenges.py`:
   - Com `p = 0`, cada desafio dá o mesmo resultado do original com a mesma seed.
   - Dealer trapaceiro com `p = 1`: em 20 mil mãos, a taxa de estouro do dealer é menor que no original.
   - Empate é derrota com `p = 1`: nenhum `game_reward` é 0.

**Experimento sugerido:** avaliar os modelos treinados no `original` com `p ∈ {0; 0,1; 0,25; 0,5}` em cada desafio, para PPO, DQN e basic strategy. O gráfico de retorno × `p` mostra qual agente é mais robusto.

**Fora do escopo:** treinar no ambiente com desafio (dá para fazer depois, registrando o desafio em `ENVS`).

**Pronto quando:** o `pytest` passa e `evaluate.py --challenge cheating_dealer --p 0.25` roda num modelo do `original`.

---

## Tarefa F — Contar cartas sem ser pego

**Ideia:** o agente escolhe quanto apostar antes de cada mão e pode ganhar dinheiro apostando alto quando a contagem está favorável. Só que a casa vigia: se as apostas acompanharem a contagem de forma óbvia, o agente é expulso e perde todas as mãos que ainda jogaria. Ele precisa aprender a explorar a contagem **e** a disfarçar.

**Hipótese:** o agente aprende a variar a aposta com a contagem, mas menos do que um contador "ingênuo", e às vezes aposta alto com contagem ruim para disfarçar. Ele termina a sessão com lucro maior que (a) apostar sempre o mínimo e (b) seguir uma regra fixa de contador que é expulsa cedo.

### Etapa 1 — Verificar que contar vale a pena (fazer antes de tudo)

O `Blackjack-v1` não tem pagamento 3:2 no blackjack natural, double down nem split. Por isso a casa tem vantagem de ~5% por mão, e é possível que nem com a contagem mais alta o jogador tenha vantagem. Se o jogo nunca for lucrativo, a melhor política é não apostar nunca, e o experimento não mostra nada.

1. Usando o `Shoe` da Tarefa D e a basic strategy da Tarefa 0, simular 1 milhão de mãos e medir o retorno médio **por true count** (de -5 a +5).
2. Se nenhum true count der retorno positivo, deixar o jogo mais favorável, nesta ordem, até algum dar: `natural=True` (paga 3:2), penetração maior (85%), menos baralhos (2).
3. Registrar no resumo final a tabela retorno × true count e as regras escolhidas. **Se nada der retorno positivo, parar aqui e reportar.**

### Etapa 2 — Ambiente

**Sessão:** um episódio é uma sessão no cassino: várias mãos seguidas no mesmo sapato (que reembaralha normalmente). A sessão acaba quando o agente é expulso ou depois de `max_hands` mãos (padrão 300).

**Duas fases por mão, com um único `Discrete(5)`** (o SB3 exige um só espaço de ações):
- **Fase de aposta:** a ação escolhe a aposta em `bets = [0, 1, 2, 4, 8]`. Aposta 0 = ficar de fora desta mão (o sapato anda mesmo assim, como se outro jogador jogasse).
- **Fase de jogo:** ação par (0, 2, 4) = parar, ação ímpar (1, 3) = pedir carta. A divisão par/ímpar evita que o agente comece favorecendo uma das duas ações.

**Recompensa:** `aposta × resultado da mão` no passo final de cada mão; 0 nos demais passos. Ser expulso encerra a sessão; o custo é perder as mãos restantes. Parâmetro `caught_penalty` (padrão 0) para testar uma punição extra.

**Vigilância da casa** (valores iniciais, calibrar na Etapa 3):
- A casa guarda as últimas `window` mãos (padrão 30) com a aposta e o true count de cada uma.
- A partir de 10 mãos jogadas, calcula a correlação entre aposta e true count nessa janela. Se as apostas forem todas iguais, a correlação é 0.
- A cada mão, a chance de expulsão é `min(1, rate × max(0, corr − threshold))`, com `threshold = 0,3` e `rate = 0,2`. Exemplo: correlação 0,8 → 10% de chance por mão.

**Observação:** `(soma do jogador, carta do dealer, ás usável, true_count, fase, suspeita)`.
- Na fase de aposta, soma e carta do dealer valem 0.
- `suspeita` é a correlação atual dividida em 5 faixas (`Discrete(5)`).
- Parâmetro `show_suspicion` (padrão `True`). Com `False`, o agente não vê a suspeita e tem que inferir o risco sozinho (variante mais difícil).

**`info`:** `game_reward` (aposta × resultado) no fim de cada mão, `true_count`, `suspicion`, `bet` e `caught` (bool) em todo passo.

**Entregas**
1. `blackjack/casino.py` com a subclasse e `make_casino_env(show_suspicion=True, **kwargs)`, com `FlattenObservation`.
2. Registro de **dois** ambientes em `ENVS`: `"casino"` e `"casino_hidden"` (`show_suspicion=False`).
3. Políticas de referência em `blackjack/casino.py`, para comparar na avaliação:
   - `flat_bettor`: aposta sempre 1 e joga basic strategy.
   - `naive_counter`: basic strategy + aposta 1, 2, 4 ou 8 conforme o true count passa de ≤1, 2, 3, ≥4.
4. `tests/test_casino.py`:
   - A fase alterna certo: aposta → jogo → … → aposta.
   - Aposta 0 não muda a recompensa, mas o sapato anda.
   - `flat_bettor` nunca é expulso (correlação 0).
   - `naive_counter` em 50 sessões é expulso em pelo menos metade delas.
   - Com `rate = 0`, ninguém é expulso.

### Etapa 3 — Calibrar a vigilância

Rodar `flat_bettor` e `naive_counter` em 1000 sessões cada. A vigilância está boa quando:
- `naive_counter` sem vigilância (`rate = 0`) lucra mais que `flat_bettor`;
- `naive_counter` com vigilância é expulso, em média, antes da metade da sessão;
- `flat_bettor` nunca é expulso.

Ajustar `threshold` e `rate` até isso valer e registrar os valores finais no resumo.

**Fora do escopo:** double down, split, treinar o agente (fica para o pipeline da Tarefa 0).

**Pronto quando:** a Etapa 1 mostra algum true count lucrativo, o `pytest` passa e a calibração da Etapa 3 está registrada.

**Atenção para o treino:** as sessões têm centenas de passos e o custo de ser expulso aparece muito depois da aposta que causou a suspeita. Vale treinar com `gamma` alto (0,995 ou mais) e mais passos que nos outros ambientes.

---

## Depois que tudo estiver pronto

Com a Tarefa 0 e pelo menos um ambiente entregues, o próximo passo é rodar 5 seeds × {PPO, DQN} × {original, customizado} e montar os gráficos num notebook em `notebooks/`. Isso fica para depois que os branches forem revisados e juntados na `main`.
