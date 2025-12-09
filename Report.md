# Midterm Project: Assessing Mathematics Learning in Higher Education

## I. Introduction
This research analyzes the "Assessing Mathematics Learning in Higher Education" dataset from the UCI Machine Learning Repository. The goal is to find patterns in student responses using clustering and forecast student performance using regression analysis.

## II. Data Discovery
- **Dataset**: 9546 records, 8 columns.
- **Features**: Student ID, Country, Question ID, Type of Answer, Question Level, Topic, Subtopic, Keywords.
- **Findings**:
    - No missing values.
    - Outliers detected in Question ID (IDs > ~1117 are rare).
    - "Basic" question level is heavily biased.
    - Most answers are incorrect (Type of Answer = 0).

## III. Methodology

### 1. Clustering Analysis (K-Means)
- **Feature Selection**: Student Country, Question Level, Topic, Type of Answer.
- **Preprocessing**: One-Hot Encoding for categorical variables, StandardScaler for numerical.
- **Dimensionality Reduction**: TruncatedSVD to reduce to 2 components for visualization.
- **Algorithm**: K-Means clustering.
- **Optimal Clusters**: 8 (based on Silhouette Score).

### 2. Linear Regression Modeling
- **Target**: Type of Answer (0 or 1).
- **Predictors**: Question ID, Student Country, Question Level, Topic.
- **Results**:
    - **R² Score**: ~0.0158 (Explains only ~1.6% of variance).
    - **RMSE**: ~0.495.
- **Interpretation**: Linear regression is not suitable for this binary classification task; the relationship is likely non-linear.

## IV. Conclusion
- Clustering successfully identified 8 distinct groups of student/question interactions.
- Linear regression failed to capture the complexity of the data.
- **Future Work**: Implement classification models (Random Forest, Logistic Regression) and use NLP for keyword analysis.
