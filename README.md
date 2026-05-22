# NeuroTelepathy: Real-Time BCI Motor Decoding Pipeline

**NeuroTelepathy** is a computational neuroscience and machine learning pipeline that simulates motor cortex spike trains and trains a recurrent neural network to decode those spikes into continuous 2D kinematic trajectories (cursor control).

This project was built to demonstrate full-stack BCI engineering: from simulating biological encoding (Georgopoulos Cosine Tuning) to deploying a time-series Neural Decoder (LSTM) in PyTorch.

---

## 🧠 System Architecture

### 1. Biological Simulation (The Motor Cortex)
* **Spike Generation:** Simulates a 96-channel intracortical microelectrode array.
* **Encoding Model:** Uses the standard Georgopoulos Cosine Tuning model, where neuron firing rates are modulated by movement direction and speed.
* **Stochasticity:** Spikes are sampled using a Poisson process to mimic the discrete, noisy nature of real action potentials.

### 2. Neural Decoder (The ML Pipeline)
* **Windowing:** The continuous spike trains are binned into 50ms intervals. The network looks at a rolling history window of 500ms (10 bins) to predict the current intended velocity.
* **Architecture:** A PyTorch Long Short-Term Memory (LSTM) network processes the temporal sequence of spikes, followed by a linear projection head to output X and Y velocities.
* **Performance:** The LSTM achieves a **Pearson Correlation (R) of >0.98** on unseen test data for both X and Y dimensions.

---

## 🛠️ Tech Stack
* **Machine Learning:** `PyTorch` (Optimized for Apple Silicon `mps`)
* **Signal Processing:** `numpy`, `scipy`
* **Visualization:** `matplotlib`

---

## 🚀 How to Run

### 1. Environment Setup
```bash
conda create -n neuro_bci python=3.11 -y
conda activate neuro_bci
pip install numpy torch matplotlib