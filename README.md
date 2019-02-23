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
- F<sub>1</sub> Score (*Harmonic Mean*) = 2\*Precision\*Recall / (Precision + Recall)
- F<sub>β</sub> Score. The smaller the β, the closer the score is to Precision. The higher the β, the closer the score is to Recall
