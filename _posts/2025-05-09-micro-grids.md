---
title: "Detection of False Data Injection Attacks in Microgrids | LSTM"
date: 2025-12-15 00:00:00 +0800
categories: [AI | Research]
tags: [machine-learning, lstm, cybersecurity, microgrids, fdia, power-systems]
description: "Deep learning approach for detecting False Data Injection Attacks in grid-connected and islanded microgrid operations."
math: true
---

> **KEYWORDS** <br>
> False Data Injection Attacks, LSTM, Microgrid Security, IEEE 33-Bus System, Anomaly Detection
{: .prompt-info }

---

## Introduction

Microgrids integrate distributed energy resources and enable autonomous operation[^1], but this creates vulnerabilities to cyber-physical attacks. False Data Injection Attacks (FDIA) manipulate sensor measurements to bypass traditional bad data detection while compromising state estimation and grid stability[^2].

This work implements an LSTM-based detection system for identifying FDIA in both grid-connected and islanded microgrid modes using time-series power flow analysis.

---

## Problem Statement

Traditional bad data detection relies on residual-based statistical methods that assume Gaussian noise. FDIA are specifically designed to evade these detectors by maintaining consistency with physical power flow constraints[^2]. The challenge is detecting coordinated attacks that appear statistically normal while causing operational harm.

Additional complexity arises from dual operating modes[^1]:
- Grid-connected mode with utility interconnection
- Islanded mode with local generation only

Each mode exhibits distinct power flow characteristics and vulnerability profiles.

---

## Methodology

### System Architecture

The detection framework consists of two components:

**MATLAB Simulation Engine**

Uses MATPOWER with a modified IEEE 33-bus distribution system[^4] configured as a microgrid:
- 33 buses with residential and commercial loads
- 4 generators: 1 slack bus + 3 DERs at buses 6, 18, and 25
- Base MVA: 5 MVA
- Voltage level: 12.66 kV
- Point of Common Coupling (PCC) at bus 34

**Python ML Pipeline**

Implements a 3-layer stacked LSTM architecture:

```python
model = keras.models.Sequential([
    keras.layers.LSTM(64, activation="tanh", return_sequences=True),
    keras.layers.Dense(units=64, activation='linear'),
    keras.layers.LSTM(64, activation="tanh", return_sequences=True),
    keras.layers.Dense(units=64, activation='linear'),
    keras.layers.LSTM(64, activation="tanh", return_sequences=True),
    keras.layers.Dense(units=64, activation='linear'),
    keras.layers.Dense(25)
])
```

Training configuration:
- Window size: 3 timesteps input, 1 timestep prediction
- Optimizer: Adam (learning rate = 0.01)
- Loss: Mean Squared Error
- Data split: 70% train, 20% validation, 10% test
- Early stopping with patience = 2 epochs

---

## Mathematical Formulation

### Attack Model

Power system measurements follow:

$$
\mathbf{z} = h(\mathbf{x}) + \mathbf{e}
$$

where $$ \mathbf{z} \in \mathbb{R}^m $$ is the measurement vector, $$ \mathbf{x} \in \mathbb{R}^n $$ is the state vector, $$ h(\cdot) $$ is the nonlinear measurement function, and $$ \mathbf{e} \sim \mathcal{N}(0, \mathbf{R}) $$ is Gaussian noise.

An FDIA injects malicious vector $$ \mathbf{a} $$:

$$
\mathbf{z}_a = h(\mathbf{x}) + \mathbf{e} + \mathbf{a}
$$

A stealthy attack satisfies $$ \mathbf{a} = \mathbf{H} \mathbf{c} $$ where $$ \mathbf{H} $$ is the Jacobian matrix.

### LSTM Detection

For time-series window $$ \mathbf{Z}_t = [\mathbf{z}_{t-w+1}, ..., \mathbf{z}_t] $$, the LSTM cell updates:

$$
\begin{aligned}
\mathbf{f}_t &= \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{z}_t] + \mathbf{b}_f) \\
\mathbf{i}_t &= \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{z}_t] + \mathbf{b}_i) \\
\tilde{\mathbf{C}}_t &= \tanh(\mathbf{W}_C \cdot [\mathbf{h}_{t-1}, \mathbf{z}_t] + \mathbf{b}_C) \\
\mathbf{C}_t &= \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t \\
\mathbf{o}_t &= \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{z}_t] + \mathbf{b}_o) \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh(\mathbf{C}_t)
\end{aligned}
$$

Prediction and anomaly score:

$$
\hat{\mathbf{z}}_{t+1} = \mathbf{W}_h \mathbf{h}_t + \mathbf{b}_h
$$

$$
\text{MSE}_t = \frac{1}{m} \sum_{i=1}^{m} (\mathbf{z}_{t+1}^{(i)} - \hat{\mathbf{z}}_{t+1}^{(i)})^2
$$

Detection rule: Attack detected if $$ \text{MSE}_t > \tau $$ where $$ \tau = 20 $$.

---

## Experimental Results

### Dataset

Two datasets generated via MATLAB simulation:
- Grid-connected mode: PCC closed, utility grid connected
- Islanded mode: PCC open, DERs provide all generation
- 25 measurement features per timestep
- Labeled normal and attack scenarios

### Performance Metrics

| Metric | Grid-Connected | Islanded |
|--------|---------------:|----------:|
| Accuracy (%) | 94.23 | 92.87 |
| Precision (%) | 91.45 | 89.32 |
| Recall (%) | 96.78 | 94.56 |
| F1 Score (%) | 94.03 | 91.87 |
| Mean MSE (Normal) | 0.12 | 0.15 |
| Mean MSE (Attack) | 127.34 | 142.67 |

### Analysis

The LSTM detector achieves over 92% accuracy in both operating modes. Attack scenarios produce MSE values approximately 1000 times higher than normal operation, providing clear separation for threshold-based detection. Grid-connected mode shows marginally better performance due to more stable power flow patterns.

The comparative analysis script generates visualizations including MSE distributions, time-series plots, performance metrics, and confusion matrices for both modes.

---

## Discussion

### Advantages

The LSTM architecture effectively captures temporal dependencies in power flow data, outperforming traditional statistical methods[^3]. The system supports both operating modes with separate trained models, maintains low computational overhead suitable for real-time deployment, and provides interpretable MSE-based anomaly scores.

### Limitations

The approach requires labeled attack data for supervised training and may not generalize to novel attack types. The fixed threshold requires tuning for different operational contexts. LSTM training is computationally intensive, though inference is efficient. Separate models for each mode necessitate accurate mode detection. The system does not address adversarial attacks targeting the detector itself.

### Future Work

Planned enhancements include testing coordinated multi-bus attacks and timing-based attacks during mode transitions. Model optimization will explore GRU and Transformer architectures, quantization for edge deployment, and federated learning for privacy preservation. Benchmarking against Random Forest, SVM, and Autoencoder baselines will establish comparative performance. Development of real-time visualization dashboards and integration with SCADA systems will enable operational deployment.

---

## Conclusion

This work demonstrates that LSTM neural networks provide effective detection of False Data Injection Attacks in microgrids operating in both grid-connected and islanded modes. The system achieves high accuracy with low false positive rates while maintaining computational efficiency suitable for real-time deployment.

As microgrids become critical infrastructure for renewable energy integration[^1], robust cybersecurity measures are essential. The combination of physics-based simulation and data-driven learning provides a practical framework for protecting smart grids against sophisticated cyber-physical attacks[^3].

---

## References

[^1]: [S. Parhizi et al., "State of the Art in Research on Microgrids: A Review," *IEEE Access*, vol. 3, pp. 890-925, 2015.](https://www.researchgate.net/publication/279232925_State_of_the_Art_in_Research_on_Microgrids_A_Review)

[^2]: [Y. Liu et al., "False Data Injection Attacks Against State Estimation in Electric Power Grids," *ACM TISSEC*, vol. 14, no. 1, 2011.](https://www.researchgate.net/publication/221608912_False_Data_Injection_Attacks_Against_State_Estimation_in_Electric_Power_Grids)

[^3]: [M. Esmalifalak et al., "Detecting Stealthy False Data Injection Using Machine Learning in Smart Grid," *IEEE Systems Journal*, vol. 11, no. 3, pp. 1644-1652, 2017.](https://www.researchgate.net/publication/261796382_Detecting_Stealthy_False_Data_Injection_Using_Machine_Learning_in_Smart_Grid)

[^4]: [IEEE 33-Bus Distribution System Overview, *Emergent Mind*, 2025.](https://www.emergentmind.com/topics/ieee-33-bus-distribution-system)
