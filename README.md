# Deep_Learning_Electric_Power_Consumption_Forecasting
# ⚡ Electric Power Consumption Forecasting using LSTM & GRU

A Deep Learning project for **time series forecasting of electric power consumption** using **Recurrent Neural Networks (RNNs)**.  
The project compares **LSTM** and **GRU** architectures to predict electricity usage in multiple zones based on historical and environmental data.

---

## 📌 Project Overview

This project implements a complete **end-to-end pipeline** for electric power consumption forecasting:
- Data loading and preprocessing
- Exploratory data analysis and visualization
- Time series sequence generation
- LSTM and GRU model training
- Model evaluation and comparison
- Visualization of predictions and training behavior

The goal is to analyze the performance trade-offs between **LSTM** and **GRU** in a real-world energy forecasting task.

---

## 🗂️ Project Structure

📁 Project Structure
├── DataSet/
│   └── powerconsumption.csv
├── main.py
├── README.md



---

## 📊 Dataset Description

The dataset consists of **time series data** related to electric power consumption and meteorological variables.

### 🔹 Input Features
- 🌡️ **Temperature**
- 💧 **Humidity**
- 🌬️ **WindSpeed**
- ☀️ **GeneralDiffuseFlows**

### 🔹 Target Variables
- ⚡ **PowerConsumption_Zone1**
- ⚡ **PowerConsumption_Zone2**
- ⚡ **PowerConsumption_Zone3**

The data is:
- Parsed as datetime
- Sorted chronologically
- Normalized using **Min-Max Scaling**

---

## 🔍 Exploratory Data Analysis

The project includes:
- Pairplots to analyze feature relationships
- Time series plots for each consumption zone
- Visual inspection of consumption trends over time

These steps help identify correlations, seasonality, and overall data behavior.

---

## 🧠 Model Architecture

Two recurrent neural network architectures are implemented and compared:

### 🔷 LSTM (Long Short-Term Memory)
- Designed to capture **long-term temporal dependencies**
- More parameters and gates
- Higher computational cost
- Slightly more stable on long-term forecasting

### 🔷 GRU (Gated Recurrent Unit)
- Simpler structure with fewer gates
- Faster training time
- Comparable predictive performance
- Efficient alternative when resources are limited

### ⚙️ Common Configuration
- Sequence length: **24 timesteps**
- Hidden units: **50**
- Output layer: **Dense(3)** (one output per zone)
- Optimizer: **Adam**
- Loss function: **Mean Squared Error (MSE)**

---

## 🏋️ Model Training

- Train/Test split: **80% / 20%**
- Batch size: **64**
- Epochs: **20**
- Validation performed on test data
- Fixed random seeds for reproducibility

Training and validation losses are monitored and plotted for both models.

---

## 📏 Model Evaluation

Performance is evaluated using standard regression metrics:

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**

Metrics are computed separately for each zone, allowing a detailed comparison between LSTM and GRU.

---

## 📈 Results & Analysis

- Both models achieve **high forecasting accuracy**
- GRU converges faster due to fewer parameters
- LSTM provides slightly better stability for long-term dependencies
- MAE and MSE values are similar across models
- Visual comparison shows good alignment between predicted and actual values

---

## 📉 Visualization of Results

The project includes:
- Forecast vs. actual plots for each zone
- Training vs. validation loss curves
- Dashboard-like comparison of LSTM and GRU learning behavior

These plots provide insights into model accuracy and generalization.

---

## ⚙️ Dependencies

Ensure **Python 3.8+** is installed.

Install required libraries using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow


GRU is computationally lighter

LSTM may perform slightly better on longer dependencies

Visualization helps assess convergence and overfitting
