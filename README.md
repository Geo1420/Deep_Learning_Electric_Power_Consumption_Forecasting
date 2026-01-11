# Deep_Learning_Electric_Power_Consumption_Forecasting
This project focuses on electric power consumption forecasting using deep learning. LSTM and GRU models are implemented to predict future energy usage from historical time series data, with feature scaling, sequence modeling, performance evaluation, and visual comparison of results.
🔌 Electric Power Consumption Forecasting using LSTM & GRU

This project focuses on forecasting electric power consumption using Deep Learning models for time series analysis, specifically LSTM and GRU neural networks. The goal is to predict future energy usage across three consumption zones based on historical data and weather-related features.

📌 Project Overview

Task: Multivariate time series forecasting

Models: LSTM (Long Short-Term Memory), GRU (Gated Recurrent Unit)

Framework: TensorFlow / Keras

Output: Power consumption prediction for 3 zones

Evaluation: MSE & MAE metrics + visual comparison

The project includes data preprocessing, visualization, sequence generation, model training, evaluation, and result analysis.

📁 Project Structure
├── DataSet/
│   └── powerconsumption.csv
├── main.py / notebook.ipynb
├── README.md


DataSet/ – contains the input dataset

main.py / notebook.ipynb – full pipeline implementation

README.md – project documentation

📊 Dataset Description

The dataset contains:

Meteorological features: Temperature, Humidity, WindSpeed, GeneralDiffuseFlows

Target variables:

PowerConsumption_Zone1

PowerConsumption_Zone2

PowerConsumption_Zone3

Datetime index for proper time series ordering

⚙️ Dependencies

Make sure you have Python 3.8+ installed.

Required libraries:

pip install numpy pandas matplotlib seaborn scikit-learn tensorflow


Or create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt

▶️ How to Run the Project

Clone the repository

git clone https://github.com/your-username/energy-forecasting-lstm-gru.git
cd energy-forecasting-lstm-gru


Place the dataset

DataSet/powerconsumption.csv


Run the script

python main.py


or open and run all cells in the Jupyter Notebook.

🧠 Model Architecture
LSTM Model

Input: time sequences (24 timesteps)

1 LSTM layer (50 units)

Dense output layer (3 values – one per zone)

GRU Model

Same structure as LSTM

GRU layer instead of LSTM for efficiency comparison

Both models use:

Optimizer: Adam

Loss Function: Mean Squared Error (MSE)

📈 Evaluation & Visualization

Metrics:

Mean Squared Error (MSE)

Mean Absolute Error (MAE)

Plots:

Actual vs Predicted consumption

Training & validation loss comparison

Time series visualization for each zone

🏁 Key Outcomes

Both LSTM and GRU successfully capture temporal patterns

GRU is computationally lighter

LSTM may perform slightly better on longer dependencies

Visualization helps assess convergence and overfitting
