# Escolha do algoritmo: PPO principal, DQN comparação

*20/11/2026*

## Resumo

Usamos **PPO** como algoritmo principal porque é estável, tolera bem a recompensa aleatória do Blackjack e é o mais documentado nos tutoriais da disciplina. Usamos **DQN** como comparação porque segue outra família de algoritmo (baseado em valor e off-policy), o que torna a comparação informativa. Os demais algoritmos do Stable-Baselines3 foram descartados por não suportarem ações discretas ou por não acrescentarem nada a um problema deste tamanho.

## O que o Blackjack exige

Três características do `Blackjack-v1` definem quais algoritmos servem:

| Característica | No Blackjack | Consequência |
| --- | --- | --- |
| Ações | 2 ações discretas: parar ou pedir carta | Só servem algoritmos com suporte a ações discretas |
| Episódios | 1 a 3 jogadas por mão | Gerar experiência é barato; eficiência de amostras pesa pouco |
| Recompensa | +1, 0 ou -1 no fim da mão, com muita sorte envolvida | O algoritmo precisa aprender bem com sinal ruidoso |

A observação (soma do jogador, carta do dealer, ás usável) é convertida num vetor de 45 posições para o SB3 aceitá-la.

## Por que PPO como principal

O PPO aprende diretamente a política (que ação tomar em cada situação) e limita o quanto ela muda a cada atualização. Isso evita que uma sequência de mãos azaradas desfaça o que já foi aprendido.

- **Estável com recompensa ruidosa:** as atualizações pequenas seguram o aprendizado mesmo quando a sorte domina o resultado de uma mão.
- **Pouco sensível a hiperparâmetros:** os valores padrão do SB3 costumam funcionar, o que libera tempo para focar nas customizações do ambiente.
- **Bem documentado:** é o algoritmo mais usado nos tutoriais da disciplina.
- **Robusto a mudanças no ambiente:** se uma customização acrescentar ações (como double down) ou informação ao estado (como contagem de cartas), o PPO continua servindo sem ajustes.

O ponto fraco é que ele descarta a experiência depois de usar, então precisa de mais mãos jogadas. No Blackjack isso custa pouco: cada mão leva de 1 a 3 jogadas.

## Por que DQN como comparação

O DQN aprende de outra forma: estima o valor de cada ação em cada situação e escolhe a de maior valor. Comparar duas abordagens diferentes mostra se o efeito de uma customização vale em geral ou só para um tipo de algoritmo.

- **Família diferente do PPO:** baseado em valor e off-policy, enquanto o PPO é baseado em política e on-policy.
- **Reaproveita experiência:** guarda mãos passadas num buffer e treina com elas várias vezes.
- **Fácil de explicar:** é a versão com rede neural do Q-learning de tabela.

O ponto fraco é ser mais sensível a hiperparâmetros (taxa de exploração, taxa de aprendizado) e mais instável com recompensa ruidosa. Por isso fica como comparação, não como principal.

## Por que não os outros

| Algoritmo | Motivo do descarte |
| --- | --- |
| DDPG, SAC, TD3 | Só funcionam com ações contínuas; o Blackjack tem ações discretas |
| HER | Exige um ambiente com objetivos a alcançar, o que o Blackjack não tem |
| A2C | Funciona, mas é uma versão menos estável da mesma ideia do PPO; não traz um contraste novo |
| QR-DQN, TRPO, ARS, RecurrentPPO | Estão no `sb3-contrib` e trazem complexidade extra sem ganho claro num problema deste tamanho |

O DDPG é citado no enunciado, mas usar ele exigiria transformar "parar ou pedir carta" num número contínuo, o que é artificial e difícil de justificar.

## Como vamos comparar

PPO e DQN serão treinados no ambiente original e no customizado, com as mesmas condições.

1. **Várias seeds:** pelo menos 5 por combinação (algoritmo × ambiente), como o enunciado pede resultados estatisticamente significativos.
2. **Métricas:** retorno médio por mão e taxas de vitória, empate e derrota, medidos numa avaliação separada do treino.
3. **Curvas de aprendizado:** retorno ao longo do treino, para ver qual algoritmo aprende mais rápido.
4. **Referência:** a estratégia ótima conhecida do Blackjack (basic strategy), com retorno médio de aproximadamente -0,05 por mão no ambiente original. Ela mostra o quanto cada agente chega perto do melhor possível.

Em aberto: o número de passos de treino por execução, a definir nos primeiros testes.
