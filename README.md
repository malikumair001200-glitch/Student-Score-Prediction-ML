# 🎓 Student Exam Score Prediction using Linear Regression

A Machine Learning project built with Python to predict students' final exam scores based on their weekly study hours using **Supervised Learning (Linear Regression)**.

---

## 📌 Project Overview
In educational analytics, understanding the direct impact of study habits on academic performance helps optimize student learning strategies. This project implements a **Linear Regression** model to quantify the relationship between **weekly study hours** ($X$) and **final exam scores** ($Y$), evaluating both model fit accuracy and error metrics.

---

## 🛠️ Tech Stack & Libraries
* **Python** (Core Programming)
* **Pandas** (Data Manipulation & Loading)
* **NumPy** (Numerical Array Computations)
* **Scikit-Learn** (Machine Learning Pipeline & Metrics)
* **Matplotlib** (Data Visualization)

---

## 📐 Machine Learning Pipeline

1. **Data Ingestion:** Load structured data from `student_scores.csv`.
2. **Feature & Target Definition:**
   - **Feature Variable ($X$):** `study-hours-per-week`
   - **Target Variable ($Y$):** `Final-Exam-Score`
3. **Model Training:** Train `sklearn.linear_model.LinearRegression` on target feature mappings.
4. **Model Evaluation Metrics:**
   - **Mean Absolute Error (MAE):** Measures average absolute prediction errors.
   - **Mean Squared Error (MSE):** Quantifies variance and squared error magnitude.
   - **Root Mean Squared Error (RMSE):** Translates error metrics back to score units.
   - **$R^2$ Score (Coefficient of Determination):** Evaluates overall variance explained by the model.
5. **Data Visualization:**
   - **Distribution Histogram:** Analyzes target score frequency distribution.
   - **Regression Trend Line:** Plots actual student scores vs. predicted regression line.

---

## 💻 How to Run the Code

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/malikumair001200-glitch/Python-Zero-To-Hero.git](https://github.com/malikumair001200-glitch/Python-Zero-To-Hero.git)
   cd Python-Zero-To-Hero
