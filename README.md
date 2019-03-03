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
  from sklearn.svm import SVC
  
  # Import GridSearchCV
  from sklearn.model_selection import GridSearchCV
  
  # Select parameters
  parameters = {'kernel': ['poly', 'rbf'], 'C': [0.1, 1, 10]}
  
  # Create a scorer
  from sklearn.metrics import make_scorer
  from sklearn.metrics import f1_score
  scorer = make_scorer(f1_score)
  
  # Define the model
  clf = SVC(gamma='auto')
  
  # Create a GridSearch Object
  grid_obj = GridSearchSV(clf, parameters, scoring = scorer)
  grid_fit = grid_obj.fit(X, y)
  
  # Choose the best estimator
  best_clf = grid_fit.best_estimator_
  ```

## Linear Regression
- Absolute trick:
  - Line equation: *y = w<sub>1</sub>x + w<sub>2</sub>*
  - Point *(p,q)* is not on the line
  - Learning rate: *α*
  - Move line closer to point: *y = (w<sub>1</sub> ± αp)x + (w<sub>2</sub> ± α)*
- Square trick: 
  - Point *(p,q<sup>'</sup>)* is on the line
  - Move line closer to point: *y = (w<sub>1</sub> ± αp(q - q<sup>'</sup>))x + (w<sub>2</sub> ± α(q - q<sup>'</sup>))*
- Error functions:
  - Mean Absolute Error
  - Mean Squared Error
- Stochastic Gradient Descent: apply the squared (or absolute) trick at every point in our data one by one, and repeat this process many times
- Batch Gradient Descent: apply the squared (or absolute) trick at every point in our data all at the same time, and repeat this process many times
- Mini-batch Gradient Descent: split the dataset into roughly equal-sized subsets, and apply Batch Gradient Descent to each subset
- Linear Regression Implementation in Sklearn:
  ```python
  from sklearn.linear_model import LinearRegression
  model = LinearRegression()
  model.fit(x_values, y_values)
  ```
- Multiple Linear Regression:
  - *n* predictor variables. Equation: *y = m<sub>1</sub>x<sub>1</sub> + m<sub>2</sub>x<sub>2</sub> + ... + m<sub>n</sub>x<sub>n</sub>*
  ```python
  from sklearn.linear_model import LinearRegression
  from sklearn.datasets import load_boston
  
  # Load the data from the boston house-prices dataset
  boston_data = load_boston()
  x = boston_data['data']
  y = boston_data['target']
  
  # Make and fit the linear regression model
  model = LinearRegression().fit(x,y)
  
  # Make a prediction using the model
  sample_house = [[2.29690000e-01, 0.00000000e+00, 1.05900000e+01, 0.00000000e+00, 4.89000000e-01,
                6.32600000e+00, 5.25000000e+01, 4.35490000e+00, 4.00000000e+00, 2.77000000e+02,
                1.86000000e+01, 3.94870000e+02, 1.09700000e+01]]
  prediction = model.predict(sample_house)
  ```
- Closed Form Solution is computationally expensive when *n* is large (*n* equations, *n* unknowns). Thus, it's the reason why we use Gradient Descent. Even though Gradient Descent won't give us 100% accurate answer, it is able to give us good enough answer which fits the data pretty well. 
- Linear Regression Warnings:
  - Linear Regression works best when the data is linear
  - Linear Regression is sensitive to outliers
- Polynomial Regression
- Regularization: Take the complexity of the model into account when calculating error
  - L1 Regularization: Add absolute values of the coefficients into the error
  - L2 Regularization: Add the square of the coefficients into the error
  - Regularization punishes complexity of the model
  - λ parameter:
    - Large λ punishes Complex model → Simple model wins
    - Small λ punishes Simple model → Complex model wins
  - L1 vs. L2 Regularization:
  
  |L1 Regularization|L2 Regularization|
  |:---:|:---:|
  |Computationally Inefficient (unless data is sparse)|Computationally Efficient|
  |Sparse Outputs|Non-Sparse Outputs|
  |Feature Selection|No Feature Selection|
- Perceptron Algorithm:
  - Perceptron step:
    For a point with coordinates *(p,q)*, label *y*, and prediction given by the equation *y<sub>prediction</sub> = step(w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + b)*:
    - If the point is correctly classified, do nothing.
    - If the point is classified positive, but it has a negative label, subtract *αp, αq,* and *α* from *w<sub>1</sub>, w<sub>2</sub>,* and *b* respectively.
    - If the point is classified negative, but it has a positive label, add *αp, αq,* and *α* from *w<sub>1</sub>, w<sub>2</sub>,* and *b* respectively.
  
  ```python
  import numpy as np
  # Setting the random seed, feel free to change it and see different solutions.
  np.random.seed(42)

  def stepFunction(t):
    if t >= 0:
        return 1
    return 0

  def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

  # Perceptron trick
  # The function should receive as inputs the data X, the labels y, the weights W (as an array), and the bias b, update the weights and bias W, b, according to the perceptron algorithm, and return W and b.
  def perceptronStep(X, y, W, b, learn_rate = 0.01):
    # Fill in code
    for i in range(len(X)):
        y_hat = prediction(X[i],W,b)
        if y[i] - y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i] - y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b
    
  # This function runs the perceptron algorithm repeatedly on the dataset, and returns a few of the boundary lines obtained in the iterations, for plotting purposes. Feel free to play with the learning rate and the num_epochs, and see your results plotted below.
  def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 100):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines
  ```
