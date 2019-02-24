# Udacity Machine Learning Engineer Nanodegree

|Term|From|To|
|:---:|:---:|:---:|
|1|20 Feb 2019|8 May 2019|
|2|||

## Course Content
### Term 1
- Decision Trees
- Naive Bayes
- Gradient Descent
- Linear Regression
- Logistic Regression
- Support Vector Machines
- Neural Network
- Kernel Method
- K-means Clustering
- Hierarchical Clustering

### Term 2

## Training and Testing model
scikit-learn
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np

# Read in the data
data = np.asarray(pd.readcsv('data.csv', header = None))
X = data[:,0:2]
y = data[:,2]

# Use train test split to split data into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

# Instantiate decision tree model
model = DecisionTreeClassifier()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate the accuracy
acc = accuracy_score(y_test, y_pred)
# acc = model.score(X_test, y_test)
```

## Evaluation Metrics
- Confusion Matrix:

  ||Guessed Positive|Guessed Negative|
  |---|:---:|:---:|
  |Positive|True Positives|False Negatives|
  |Negative|False Positives|True Negatives|

  - Type 1 Error: False Positive
  - Type 2 Error: False Negative

- Accuracy = (True Positives + True Negatives)/(Total)
- Recall = True Positives / (True Positives + False Negatives)
- Precision = True Positives / (True Positives + False Positives)
- High Recall (False Positives: Okay, False Negatives: Not Okay)
- High Precision (False Positives: Not Okay, False Negatives: Okay)
- F<sub>1</sub> Score (*Harmonic Mean*) = 2 × Precision × Recall / (Precision + Recall)
- F<sub>β</sub> Score:
  - F<sub>β</sub> Score = (1 + β<sup>2</sup>) × Precision × Recall / (β × Precision + Recall)
  - If β = 0 then F<sub>β</sub> = Precision
  - If β = ∞ then F<sub>β</sub> = Recall
- Receiver Operating Characteristic (ROC Curve)
- Regression Metrics:
  - Mean Absolute Error:
  ```python
  from sklearn.metrics import mean_absolute_error
  from sklearn.linear_model import LinearRegression
  
  classifier = LinearRegression()
  classifier.fit(X, y)
  guesses = classifier.predict(X)
  error = mean_absolute_error(y, guesses)
  ```
  - Mean Squared Error:
  ```python
  from sklearn.metrics import mean_squared_error
  from sklearn.linear_model import LinearRegression
  
  classifier = LinearRegression()
  classifier.fit(X, y)
  guesses = classifier.predict(X)
  error = mean_squared_error(y, guesses)
  ```
  - R2 Score = 1 - Model_score / Simple_model_score
    - Bad model: R2 Score close to 0
    - Good model: R2 Score close to 1

## Model Selection
- Type of errors:
  - Underfitting: 
    - Error due to bias (high bias)
    - Bad on training set
    - Bad on testing set
  - Overfitting:
    - Error due to variance (high variance)
    - Great on the training set
    - Bad on testing set
- Model Complexity Graph
- Data set:
  - Training
  - Cross Validation
  - Testing
- K-fold cross validation: useful to recycle data
  - Break the data set into *k* buckets
  - Train the model *k* times
  - Average the results to get the final model
  - sklearn:
  ```python
  from sklearn.model_selection import KFold
  kf = KFold(12, 3, shuffle = True)
  ```
- Learning Curves
- Grid Search:
  ```python
  from sklearn.tree import DecisionTreeClassifier
  
  # Import GridSearchCV
  from sklearn.model_selection import GridSearchCV
  
  # Select parameters
  parameters = {'kernel': ['poly', 'rbf'], 'C': [0.1, 1, 10]}
  
  # Create a scorer
  from sklearn.metrics import make_scorer
  from sklearn.metrics import f1_score
  scorer = make_scorer(f1_score)
  
  # Define the model
  clf = DecisionTreeClassifier(random_state=42)
  
  # Create a GridSearch Object
  grid_obj = GridSearchSV(clf, parameters, scoring = scorer)
  grid_fit = grid_obj.fit(X, y)
  
  # Choose the best estimator
  best_clf = grid_fit.best_estimator_
  ```
