# 🏏 IPL Smart Analyst

An end-to-end Machine Learning web application that analyzes IPL data to generate match predictions, team insights, and player performance analytics through an interactive dashboard.

---

## 🚀 Features

### 🔮 Match Prediction
- Predicts match winner using a trained ML model
- Considers:
  - Teams
  - Toss winner & decision
  - Venue
  - Historical performance stats
- Displays win probability with confidence bar

### 📊 Team Analysis
- Compare two teams based on:
  - Total wins
  - Win percentage
  - Batting strike rate
  - Bowling economy
- Interactive visualizations using Plotly

### 🧠 Player Insights
- Top batsmen by strike rate
- Top bowlers by economy
- Adjustable filters (min balls faced/bowled)

---

## 🧠 Machine Learning

- Model: Random Forest Classifier
- Features used:
  - Team win percentage
  - Batting strike rate
  - Bowling economy
  - Toss advantage
  - Venue encoding
- Evaluation:
  - Accuracy
  - Confusion Matrix

---

## 🛠️ Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Streamlit**
- **Plotly**
- **Joblib**

