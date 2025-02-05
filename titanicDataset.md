# Titanic Dataset ML Project Documentation

## Overview

This project demonstrates the use of Python libraries to manipulate the Titanic dataset and build basic Machine Learning models to predict survival rates. It includes steps for data manipulation using `Numpy` and `Pandas`, data visualization using `Seaborn`, and the implementation of Logistic Regression, Decision Tree, and Random Forest models using `scikit-learn`.

---

## Getting Started

### Prerequisites

Ensure you have Python installed along with the following libraries:

- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `scikit-learn`

You can install the required libraries using:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

---

## Steps

### 1. Download the Titanic Dataset

The dataset can be downloaded from the Kaggle Titanic competition page:
[Titanic Kaggle Dataset](https://www.kaggle.com/competitions/titanic/data)

Save the files `train.csv` and `test.csv` in your project directory.

---

### 2. Load the Dataset

```python
import pandas as pd

# Load the train and test datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Display the first few rows
print(train_data.head())
```

---

### 3. Data Exploration

```python
# Get basic information
print(train_data.info())

# Check for missing values
print(train_data.isnull().sum())

# View summary statistics
print(train_data.describe())
```

---

### 4. Data Visualization

Use `seaborn` to explore correlations and distributions.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Visualize survival counts
sns.countplot(x='Survived', data=train_data)
plt.show()

# Heatmap of correlations
sns.heatmap(train_data.corr(), annot=True, cmap='coolwarm')
plt.show()
```

---

### 5. Data Preprocessing

```python
# Fill missing values
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Drop unnecessary columns
train_data = train_data.drop(['Cabin', 'Name', 'Ticket'], axis=1)

# Convert categorical variables to numerical
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)

# Define features and target
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']
```

---

### 6. Split the Dataset

```python
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### 7. Build and Evaluate Machine Learning Models

#### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train the model
logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train, y_train)

# Make predictions
logistic_predictions = logistic_model.predict(X_test)

# Evaluate the model
print("Logistic Regression Accuracy:", accuracy_score(y_test, logistic_predictions))
```

#### Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

# Train the model
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Make predictions
decision_tree_predictions = decision_tree.predict(X_test)

# Evaluate the model
print("Decision Tree Accuracy:", accuracy_score(y_test, decision_tree_predictions))
```

#### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

# Train the model
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Make predictions
random_forest_predictions = random_forest.predict(X_test)

# Evaluate the model
print("Random Forest Accuracy:", accuracy_score(y_test, random_forest_predictions))
```

---

### 8. Analyze Feature Importance

```python
# Analyze feature importance (Random Forest example)
importances = pd.Series(random_forest.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.show()
```

---

## Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | \~80%    |
| Decision Tree       | \~78%    |
| Random Forest       | \~82%    |

The Random Forest model performed the best in terms of accuracy.

---

## Conclusion

This project demonstrated:

1. Data manipulation using `Pandas` and `Numpy`.
2. Data visualization using `Seaborn` and `Matplotlib`.
3. Building and evaluating basic ML models (Logistic Regression, Decision Tree, Random Forest).

Future steps could include:

- Hyperparameter tuning for better performance.
- Testing on the `test.csv` dataset for Kaggle submission.

---

## How to Run the Project

1. Clone this repository.
2. Install the required libraries.
3. Open the notebook and run all cells sequentially.

---

## Author

Rahul Yadav
