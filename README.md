# Math Learning Assessment & Analysis 📊

An Educational Data Mining project applying Machine Learning techniques to analyze student performance in higher education mathematics.

## 🚀 Project Overview
This project explores the "Assessing Mathematics Learning in Higher Education" dataset to uncover patterns in student learning and predict success rates. By applying **K-Means Clustering**, we identified distinct student performance groups, and through **Linear Regression**, we evaluated the predictability of answer correctness based on question attributes.

## 🔑 Key Features
*   **Exploratory Data Analysis (EDA)**: Comprehensive statistical analysis and visualization of student demographics and question difficulty.
*   **Unsupervised Learning**: K-Means clustering (k=8) to segment student response patterns.
*   **Dimensionality Reduction**: TruncatedSVD for visualizing high-dimensional categorical data.
*   **Supervised Learning**: Linear Regression baseline for performance prediction.
*   **Pipeline Architecture**: Scikit-learn Pipelines for clean, reproducible preprocessing (OneHotEncoder, StandardScaler).

## 🛠️ Technologies
*   **Python**
*   **Pandas & NumPy** (Data Manipulation)
*   **Scikit-learn** (Clustering, Regression, Preprocessing)
*   **Matplotlib & Seaborn** (Visualization)

## 📈 Results
*   **Clustering**: Identified **8 optimal clusters** representing different student-question interaction profiles.
*   **Prediction**: Linear regression yielded a low R² (~1.6%), confirming that student success is a complex, non-linear classification problem better suited for Random Forests or Logistic Regression (suggested for future work).

## 📂 Repository Structure
*   `datamining_midterm.ipynb`: Complete Jupyter Notebook with code, visualizations, and analysis.
*   `Report.md`: Summary of methodology and findings.
