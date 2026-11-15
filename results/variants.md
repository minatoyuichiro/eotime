# 変種の比較（自動生成: compare_variants.py）

| 対象 | main (パルス / ε 閾値 / 1e-2 での漏れ) | fixed 1q (パルス / ε 閾値 / 1e-2 での漏れ) | CZ (Chadwick) (パルス / ε 閾値 / 1e-2 での漏れ) | LCCZ (Chadwick) (パルス / ε 閾値 / 1e-2 での漏れ) |
|---|---|---|---|---|
| VQE H2 | 32 / 5.9e-03 / 2.3e-03 | 59 / 3.1e-03 / 2.3e-03 | 38 / 6.2e-03 / 2.6e-03 | 56 / 3.9e-03 / 5.6e-03 |
| VQE LiH | 184 / 5.5e-03 / 8.0e-03 | 328 / 3.5e-03 / 6.5e-03 | — | — |
| QAOA C4_p2 | 498 / 7.6e-03 / 4.4e-02 | 776 / 4.4e-03 / 4.9e-02 | — | — |
| QAOA prism6_p2 | 1147 / 4.2e-03 / 8.9e-02 | 1721 / 2.5e-03 / 1.0e-01 | — | — |
| grover3 | 1216 / 5.6e-03 / 1.1e-01 | 1817 / 2.9e-03 / 1.8e-01 | 1450 / 4.9e-03 / 7.1e-02 | 2170 / 3.3e-03 / 2.3e-01 |
| qpe_phi0.375 | 503 / 9.0e-03 / 2.6e-02 | 746 / 5.1e-03 / 2.0e-02 | 561 / 9.0e-03 / 3.2e-02 | 777 / 8.6e-03 / 4.3e-02 |
| repcode_d3 | 120 / 7.6e-03 / 6.7e-03 | 208 / 5.0e-03 / 7.0e-03 | 120 / 6.0e-03 / 1.4e-02 | 192 / 4.2e-03 / 2.0e-02 |

## CNOT のゲート単位

| CNOT [main] | pulses 28 T=24.5 | eps 1e-2: infid 5.33e-03 leak 1.82e-03 (34%) | sigma_add 1e-2: infid 5.68e-03 leak 2.11e-03 |
| CNOT [fixed 1q] | pulses 46 T=37.7 | eps 1e-2: infid 1.52e-02 leak 1.82e-03 (12%) | sigma_add 1e-2: infid 9.82e-03 leak 2.23e-03 |
| CNOT [CZ (Chadwick)] | pulses 34 T=34.1 | eps 1e-2: infid 5.13e-03 leak 2.48e-03 (48%) | sigma_add 1e-2: infid 7.11e-03 leak 2.19e-03 |
| CNOT [LCCZ (Chadwick)] | pulses 52 T=49.6 | eps 1e-2: infid 1.18e-02 leak 4.13e-03 (35%) | sigma_add 1e-2: infid 1.07e-02 leak 3.97e-03 |
