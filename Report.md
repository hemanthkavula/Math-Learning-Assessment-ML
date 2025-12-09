# Midterm Project: Assessing Mathematics Learning in Higher Education

## I. Introduction
This research analyzes the "Assessing Mathematics Learning in Higher Education" dataset from the UCI Machine Learning Repository. The goal is to find patterns in student responses using clustering and forecast student performance using regression analysis.

The dataset captures student interactions with various math questions, categorized by topic, subtopic, difficulty level, country, and correctness of answers. This analysis is vital in identifying key performance drivers and improving content delivery in educational platforms.

## II. Data Discovery

### Dataset Overview
- **Records**: 9,546
- **Columns**: 8 (Student ID, Student Country, Question ID, Type of Answer, Question Level, Topic, Subtopic, Keywords)
- **Data Types**: Numerical (Student ID, Question ID, Type of Answer) and Categorical (Country, Question Level, Topic, Subtopic, Keywords)

### Key Findings
1. **No Missing Values**: Dataset is complete with no null entries.
2. **Outliers in Question ID**: Questions with IDs > 1117 appear much less frequently, suggesting high-difficulty or rare questions.
3. **Class Imbalance**: Type of Answer is heavily skewed toward 0 (incorrect responses > correct).
4. **Basic Question Bias**: The "Basic" difficulty level is over-represented in the dataset.
5. **Student Distribution**: Wide range of student IDs indicates diverse student population across countries.

## III. Methodology

### 1. Clustering Analysis

**Objective**: Group related observations by student characteristics and performance patterns.

**Feature Selection**:
- Student Country (categorical)
- Question Level (categorical)
- Topic (categorical)
- Type of Answer (numerical: 0 = incorrect, 1 = correct)

**Data Preprocessing**:
- **One-Hot Encoding**: Transformed categorical variables into binary indicators.
- **Standard Scaling**: Normalized numerical features to ensure equal weighting in distance calculations.

**Dimensionality Reduction**:
- **TruncatedSVD**: Compressed high-dimensional feature space from 50+ features to 2 components for 2D visualization while preserving maximum variance.

**Algorithm**:
- **K-Means Clustering**: Applied on reduced feature space.
- **Silhouette Score Optimization**: Tested k values from 2 to 10; optimal k = **8**.

**Results**:
- Successfully identified 8 distinct clusters representing different student-question interaction profiles.
- Clusters vary in correctness rates, indicating meaningful segmentation based on country, topic, and difficulty.

### 2. Linear Regression Modeling

**Objective**: Predict student answer correctness using question and student attributes.

**Features & Target**:
- **Predictors**: Question ID, Student Country, Question Level, Topic
- **Target**: Type of Answer (0 or 1)

**Data Preprocessing**:
- **One-Hot Encoding**: Categorical variables with drop='first' to avoid multicollinearity.
- **No Imputation**: Dataset had no missing values.
- **Train-Test Split**: 80% training, 20% testing.

**Model Evaluation**:
- **R² Score**: ~0.0158 (explains only ~1.6% of variance)
- **RMSE**: ~0.495 (average prediction error on binary scale)

**Interpretation**:
- Linear regression has **limited predictive power** due to the binary nature of the target variable and complex non-linear relationships in educational data.
- Residual distribution shows slight non-normality and heteroscedasticity, confirming linear regression's unsuitability.

## IV. Visualizations

1. **Type of Answer Distribution**: Histogram showing overwhelming bias toward incorrect answers (0).
2. **Question ID by Answer Type Boxplot**: Reveals that certain question IDs have higher incorrect answer rates.
3. **Student ID Distribution**: Histogram shows broad range of student participation.
4. **Question Level Countplot**: Basic questions dominate the dataset.
5. **Cluster Visualization**: 2D scatter plot of SVD-reduced clusters with color-coded assignments.

## V. Conclusion

### Key Takeaways
- **Clustering Success**: K-Means effectively segmented students into 8 distinct groups based on country, topic, and difficulty patterns.
- **Regression Limitations**: Linear regression is insufficient for this binary classification problem; alternative models are needed.
- **Data Insights**: Student success in mathematics is influenced by multiple factors including question difficulty, country-specific education systems, and topic complexity.

### Future Improvements
1. **Classification Models**: Implement Logistic Regression, Random Forest, or Gradient Boosting for better predictions.
2. **NLP Analysis**: Apply text mining to Keywords column for deeper insight into question content.
3. **Time-Series Analysis**: Track student improvement over multiple assessments.
4. **Aggregate Metrics**: Include student-level statistics (accuracy over time, per-topic performance).
5. **Cross-Validation**: Use k-fold cross-validation for more robust model evaluation.
6. **Hyperparameter Tuning**: Optimize clustering and classification parameters using GridSearchCV.

## VI. References
- UCI Machine Learning Repository: [Assessing Mathematics Learning in Higher Education](https://www.kaggle.com/datasets/gauravduttakiit/assessing-mathematics-learning-in-higher-education)
- Scikit-learn Documentation: Clustering and Regression Models
- Educational Data Mining: Best Practices for Analysis
