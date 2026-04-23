# 💤 Personalized Sleep Quality Predictor

A machine learning project that analyzes daily habits to predict sleep quality scores and next-day fatigue. This repository demonstrates end-to-end ML workflows including data preprocessing, feature engineering, and predictive modeling using Scikit-Learn.

## 📊 Project Overview
The goal of this project is to understand how various lifestyle factors (sleep hours, caffeine, stress, screen time) impact overall sleep quality. It features two primary models:
1. **Linear Regression:** To predict a continuous `sleep_quality_score` (0-100).
2. **Logistic Regression:** To classify whether the user will feel `tired_next_day` (Binary).

## 🛠 Tech Stack
- **Languages:** Python 3.x
- **ML Libraries:** Scikit-Learn, NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook / Google Colab

## 📁 Repository Structure
- `DATA/`: Contains the sleep habit dataset.
- `NOTEBOOK/`: Jupyter notebooks with exploratory data analysis and model experimentation.
- `plots/`: Generated visualizations from the analysis.
- `main.py`: A production-ready script to train models and make predictions.
- `requirements.txt`: List of dependencies.

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/amazephoenix-bit/sleep-quality-predictor.git
   cd sleep-quality-predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the prediction script:**
   ```bash
   python main.py
   ```

## 🔮 Future Improvements
- Transitioning to Pytorch for more complex sequence modeling (e.g., analyzing sleep patterns over time).
- Integrating a web dashboard for real-time data logging and visualization.

---
*Developed as part of my AI/ML learning journey at Kristu Jayanti University.*