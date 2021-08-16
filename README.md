# Poisoning the Search Space in Neural Architecture Search.

This repository contains the poisoning experiments carried out on the [ENAS algorithm](https://github.com/rusbridger/ENAS-Experiments).

| Group                   | Poisoning Set                                       |
| ----------------------- | --------------------------------------------------- |
| Baseline                | P0 = {}                                             |
| Identity                | P1 = Identity                                       |
| Gaussian Noise          | P2 = Gaussian(sigma=2) **or** Gaussian(sigma=10)    |
| Dropout                 | P3 = Dropout(p=0.9) **or** Dropout(p=1.0)           |
| Transposed Convolutions | P4 = 3x3 TransposedConv **and** 5x5 Transposed Conv |
| Bad Convolutions        | P5 = Conv(kernel=3, padding=50, dilation=50)        |
| Grouped Operations      | P6 = P1 + P2 + P3 + P4 + P5                         |
