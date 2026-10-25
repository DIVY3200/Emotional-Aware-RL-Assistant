## Known user types

| Policy | Mean return (95% CI) | Mean valence | Speaks | CI over |
|---|---|---|---|---|
| PPO told the user type (upper bound) | +5.63 ± 0.06 | +0.302 | 55% | training seeds |
| Greedy oracle (sees hidden state) | +4.78 ± 0.22 | +0.276 | 75% | episodes |
| PPO (ours) | +3.66 ± 0.19 | +0.201 | 55% | training seeds |
| PPO + 4-turn memory | +3.42 ± 0.26 | +0.192 | 56% | training seeds |
| DQN + memory | +3.37 ± 0.12 | +0.188 | 55% | training seeds |
| A2C + memory | +3.22 ± 0.18 | +0.178 | 51% | training seeds |
| PPO, top-1 label input | +3.11 ± 0.19 | +0.177 | 57% | training seeds |
| Recurrent PPO (LSTM) | +3.00 ± 0.33 | +0.169 | 51% | training seeds |
| Thompson sampling (bandit) | +2.28 ± 0.10 | +0.144 | 75% | training seeds |
| Rule table + 2-turn cooldown | +2.21 ± 0.26 | +0.121 | 35% | episodes |
| LinUCB (contextual bandit) | +2.19 ± 0.62 | +0.139 | 71% | training seeds |
| Always listen | +0.74 ± 0.26 | +0.032 | 0% | episodes |
| Random | -1.55 ± 0.28 | -0.024 | 83% | episodes |
| v1 rule table | -4.68 ± 0.25 | -0.146 | 100% | episodes |
| Always comfort | -6.74 ± 0.24 | -0.189 | 100% | episodes |

## Unseen blended users

| Policy | Mean return (95% CI) | Mean valence | Speaks | CI over |
|---|---|---|---|---|
| PPO told the user type (upper bound) | +4.35 ± 0.06 | +0.235 | 54% | training seeds |
| Greedy oracle (sees hidden state) | +4.05 ± 0.23 | +0.234 | 74% | episodes |
| PPO (ours) | +3.74 ± 0.10 | +0.204 | 55% | training seeds |
| PPO + 4-turn memory | +3.53 ± 0.18 | +0.196 | 56% | training seeds |
| DQN + memory | +3.41 ± 0.15 | +0.189 | 55% | training seeds |
| A2C + memory | +3.32 ± 0.22 | +0.182 | 51% | training seeds |
| PPO, top-1 label input | +3.15 ± 0.08 | +0.178 | 57% | training seeds |
| Recurrent PPO (LSTM) | +2.97 ± 0.32 | +0.166 | 51% | training seeds |
| LinUCB (contextual bandit) | +2.10 ± 0.85 | +0.134 | 71% | training seeds |
| Thompson sampling (bandit) | +2.03 ± 0.11 | +0.132 | 75% | training seeds |
| Rule table + 2-turn cooldown | +1.96 ± 0.25 | +0.108 | 35% | episodes |
| Always listen | +0.63 ± 0.26 | +0.026 | 0% | episodes |
| Random | -2.07 ± 0.26 | -0.049 | 84% | episodes |
| v1 rule table | -4.80 ± 0.26 | -0.153 | 100% | episodes |
| Always comfort | -6.73 ± 0.24 | -0.189 | 100% | episodes |

## Paired test vs PPO (ours)

| Policy | Δ return (ours − policy) | 95% CI | Wilcoxon p |
|---|---|---|---|
| PPO told the user type (upper bound) | -1.97 | ± 0.17 | 1.7e-83 |
| Greedy oracle (sees hidden state) | -1.12 | ± 0.22 | 3.7e-24 |
| PPO + 4-turn memory | +0.24 | ± 0.12 | 3.7e-04 |
| DQN + memory | +0.29 | ± 0.13 | 1.4e-05 |
| A2C + memory | +0.44 | ± 0.14 | 7.9e-10 |
| PPO, top-1 label input | +0.55 | ± 0.14 | 8.2e-14 |
| Recurrent PPO (LSTM) | +0.66 | ± 0.15 | 1.2e-17 |
| Thompson sampling (bandit) | +1.38 | ± 0.21 | 1.7e-29 |
| Rule table + 2-turn cooldown | +1.45 | ± 0.24 | 3.7e-28 |
| LinUCB (contextual bandit) | +1.47 | ± 0.18 | 3.0e-47 |
| Always listen | +2.92 | ± 0.25 | 1.7e-87 |
| Random | +5.21 | ± 0.27 | 8.2e-142 |
| v1 rule table | +8.34 | ± 0.26 | 6.1e-164 |
| Always comfort | +10.40 | ± 0.27 | 5.0e-165 |
