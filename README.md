# Salary Prediction

## Introduction
This project aims to predict salaries based on various factors, such as age, gender, education level, job title, and years of experience. We have used a dataset containing 6704 rows and 6 columns to develop and evaluate our salary prediction model.

## Data Preprocessing

### Handling Missing Values
We checked for missing values in the dataset and removed rows with missing data, ensuring a clean dataset for modeling.

## Data Visualization

### Age Distribution
![Age Distribution](images/age_distribution.png)
*This histogram shows the distribution of ages in the dataset.*

### Gender Distribution
![Gender Distribution](images/gender_distribution.png)
*A pie chart representing the gender distribution among the dataset.*

### Salary vs. Education
![Salary vs. Education](images/salary_education.png)
*A boxplot displaying the relationship between education level and salary.*

### Correlation Heatmap
![Correlation Heatmap](images/correlation_heatmap.png)
*A heatmap illustrating the correlation between different features and salary.*

## Model Building and Evaluation

### Model Selection
We explored various machine learning algorithms, including Linear Regression, Decision Trees, and Random Forests, to build our salary prediction model. Hyperparameter tuning was performed using GridSearchCV to find the best model configuration.

### Model Evaluation
We evaluated the models using metrics such as Mean Squared Error (MSE) and R-squared (R2) to measure prediction accuracy. The best-performing model achieved an MSE of X and an R2 score of Y on the test set.

### Feature Importance
![Feature Importance](images/feature_importance.png)
*A bar chart depicting the importance of different features in predicting salary.*

## Conclusion

In conclusion, our salary prediction model, trained on a well-preprocessed dataset, successfully predicts salaries based on various factors. This project demonstrates the importance of data preprocessing, feature engineering, and model selection in creating an accurate predictive model.

## Usage

To use our salary prediction model, you can follow these steps:

1. Clone this repository.
2. Install the required libraries listed in the `requirements.txt` file.
3. Run the provided Jupyter notebook or Python script to load the model and make predictions on new data.

## References

- Dataset source: [Link to Dataset](https://example.com/dataset)
- Scikit-Learn: [https://scikit-learn.org](https://scikit-learn.org)

