# Process-Sensor-Anomaly-Detection
### Tennessee Eastman Process — Isolation Forest & Autoencoder

[Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
[TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
[scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green.svg)
[Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)


##  Project Overview

This project applies unsupervised machine learning to detect process faults in a simulated chemical plant — the **Tennessee Eastman Process (TEP)**. Two anomaly detection approaches are implemented and compared:

- **Isolation Forest** — tree-based unsupervised outlier detection
- **Autoencoder Neural Network** — reconstruction-based deep learning approach

The goal mirrors real-world industrial challenges: identify process faults early using multivariate sensor data, without relying on labelled fault examples during training.

## The Tennessee Eastman Process

The TEP is a widely used benchmark simulation of a chemical plant with 5 unit operations:

| Unit | Description |
|------|-------------|
| Reactor | Exothermic gas-phase reaction |
| Condenser | Product cooling |
| Compressor | Recycle gas compression |
| Separator | Gas-liquid separation |
| Stripper | Product purification |

**Dataset characteristics:**
- 52 process variables (41 sensor measurements + 11 manipulated variables)
- 21 distinct fault types (feed disturbances, valve failures, sensor faults)
- Sampled every 3 minutes of simulated plant time
- 500 simulation runs per condition


##  Methods

### 1. Isolation Forest
An ensemble of random decision trees that isolates anomalies by exploiting the fact that faults are rare and different — requiring far fewer splits to isolate than normal operating points.

- Trained exclusively on normal operating data
- No assumption about data distribution
- Anomaly score based on average path length across trees

### 2. Autoencoder Neural Network
A neural network trained to compress and reconstruct normal sensor data through a bottleneck layer. Faults produce high reconstruction error because the network has never seen them.

**Architecture:** 52 → 32 → 16 → 8 → 16 → 32 → 52

- Encoder learns a compact representation of normal plant behavior
- Decoder reconstructs original sensor readings from compressed form
- Anomaly score = Mean Squared Reconstruction Error (MSE)
- Threshold set at 95th percentile of training reconstruction errors


##  Results

### Model Comparison — Fault 1 (Feed Composition Step Change)

| Metric | Isolation Forest | Autoencoder |
|--------|-----------------|-------------|
| Accuracy | ~50% | **95%** |
| Precision | 0.08 | **0.94** |
| Recall | 0.06 | **0.95** |
| F1-Score | 0.07 | **0.95** |
| Mean Error — Normal | — | 0.52 |
| Mean Error — Fault | — | 8.47 |

The Autoencoder outperforms Isolation Forest significantly because it learns **multivariate sensor relationships** rather than treating each sensor independently.

### Key Finding
The Autoencoder reconstruction error increases **16x** between normal and faulty conditions — enabling reliable early fault detection with the threshold set from normal operating data alone.


## Project Structure


process-sensor-anomaly-detection/

├── Process_Sensor_Anomaly_Detection.ipynb   (Main notebook)
├── README.md                                 (This file)

└── data/                                     
    ├── TEP_FaultFree_Training.RData
    ├── TEP_FaultFree_Testing.RData
    ├── TEP_Faulty_Training.RData
    └── TEP_Faulty_Testing.RData





---

##  Key Learnings

- **Data scaling is critical** — neural networks require StandardScaler to normalize sensor ranges before training
- **Contamination parameter matters** — Isolation Forest performance is sensitive to the assumed anomaly rate
- **More training data reduces overfitting** — using 5 simulation runs vs 1 dramatically improved Autoencoder generalization
- **Threshold selection is a design choice** — the 95th percentile approach balances false alarms vs missed detections


## 🔮 Future Work

- [ ] Evaluate across all 21 fault types, not just Fault 1
- [ ] Implement LSTM Autoencoder to exploit temporal sensor patterns
- [ ] Add feature importance analysis — which sensors contribute most to fault detection?
- [ ] Deploy as a real-time monitoring dashboard

---

##  Author

**Bhagwatiben Dayal**
M.Sc. Chemical and Energy Engineering
Otto-von-Guericke-Universität Magdeburg


##  References

- Downs, J.J. & Vogel, E.F. (1993). A plant-wide industrial process control problem. *Computers & Chemical Engineering*, 17(3), 245–255.
- Dataset: [csafta on Kaggle](https://www.kaggle.com/datasets/csafta/tennessee-eastman-process)



