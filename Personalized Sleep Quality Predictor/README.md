# 💤 Personalized Sleep Quality Predictor

## 🔍 Project Overview
This project predicts:
- A **sleep quality score** (0–100), and
- Whether I will feel **tired the next day** (0 or 1),

based on my daily habits like sleep hours, screen time before bed, caffeine intake, stress level, exercise, steps, and bedtime category.

## 📊 Dataset
- File: `data/sleep_data.csv`
- Each row = one night of sleep.
- Columns include:
  - `sleep_hours`
  - `screen_time_before_bed`
  - `caffeine_mg`
  - `stress_level`
  - `exercise_minutes`
  - `steps`
  - `bedtime_category`
  - `sleep_quality_score`
  - `tired_next_day`

## 🧠 Models Used
- **Linear Regression** → predicts `sleep_quality_score`
- **Logistic Regression** → predicts `tired_next_day`

## 🛠 Tech Stack
- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

## 🚀 How to Run
1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
