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
  grid_obj = GridSearchCV(clf, parameters, scoring = scorer)
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
  - Linear Regression works best when the data is linear.
  - Linear Regression is sensitive to outliers.
- Polynomial Regression
- Regularization: Take the complexity of the model into account when calculating error.
  - L1 Regularization: Add absolute values of the coefficients into the error.
  - L2 Regularization: Add the square of the coefficients into the error.
  - Regularization punishes complexity of the model.
  - λ parameter:
    - Large λ punishes Complex model → Simple model wins
    - Small λ punishes Simple model → Complex model wins
  - L1 vs. L2 Regularization:
  
  |L1 Regularization|L2 Regularization|
  |:---:|:---:|
  |Computationally Inefficient (unless data is sparse)|Computationally Efficient|
  |Sparse Outputs|Non-Sparse Outputs|
  |Feature Selection|No Feature Selection|

## Perceptron Algorithm
- Perceptron step: For a point with coordinates *(p,q)*, label *y*, and prediction given by the equation *y<sub>prediction</sub> = step(w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + b)*:
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
  # The function should receive as inputs the data X, the labels y, 
  # the weights W (as an array), and the bias b, update the weights 
  # and bias W, b, according to the perceptron algorithm, and return W and b.
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
    
  # This function runs the perceptron algorithm repeatedly on the dataset, 
  # and returns a few of the boundary lines obtained in the iterations, 
  # for plotting purposes. 
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

## Decision Trees
- Entropy:
  - High Knowledge → Low Entropy
  - Medium Knowledge → Medium Entropy
  - Low Knowledge → High Entropy
  - Entropy = -Σ*p<sub>i</sub>×log<sub>2</sub>(p<sub>i</sub>)*
  - Information Gain = Entropy(Parent) - 0.5×\[Entropy(Child<sub>1</sub>) + Entropy(Child<sub>2</sub>)\]
  - Maximize Information Gain at each step.
- Decision Tree tends to overfit.
- Hyperparameters for decision tree:
  - `max_depth` is the largest possible length from the root to a leaf.
  - `min_samples_split`: a node must have at least `min_samples_split` samples in order to be large enough to be split. If it has less than `min_samples_split` samples, it will not be split, and the splitting process stops. `min_samples_split` doesn't control the minimum size of a leaf.
  - `min_samples_leaf` is the minimum number of samples that a leaf must have. `min_samples_leaf` can be a float or an integer.
- Large `max_depth` very often causes overfitting, since a tree that is too deep, can memorize the data. Small `max_depth` can result in a very simple model, which may cause underfitting.
- Small `min_samples_split` may result in a complicated, highly branched tree, which can mean the model has memorized the data, or in other words, overfit. Large `min_samples_split` may result in the tree not having enough flexibility to get built, and may result in underfitting.
- Decision Tree in sklearn:
    ```python
    # Import statements 
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score
    import pandas as pd
    import numpy as np

    # Read the data.
    data = np.asarray(pd.read_csv('data.csv', header=None))
    # Assign the features to the variable X, and the labels to the variable y. 
    X = data[:,0:2]
    y = data[:,2]

    # Creat and fit the model
    model = DecisionTreeClassifier().fit(X, y)

    # Make predictions. Store them in the variable y_pred.
    y_pred = model.predict(X)

    # Calculate the accuracy and assign it to the variable acc.
    acc = accuracy_score(y, y_pred)
    ```
- One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction. The problem with label encoding is that it assumes the higher the categorical value, the better the category is.

## Naive Bayes
- Naive assumption: events are independent. *P(A AND B) = P(A)×P(B)*
- Implementation in sklearn:
  ```python
  from sklearn.naive_bayes import MultinomialNB
  naive_bayes = MultinomialNB()
  naive_bayes.fit(training_data, y_train)
  ```
- Advantages of Naive Bayes:
  - Ability to handle an extremely large number of features. Performs well even with the presence of irrelevant features and is relatively unaffected by them. 
  - Relative simplicity. Naive Bayes works well right out of the box and tuning it's parameters is rarely ever necessary, except usually in cases where the distribution of the data is known. It rarely ever overfits the data. 
  - Model training and prediction times are very fast for the amount of data it can handle. 
- Bag of Words(BoW) concept specifies the problems that have a collection of text data that needs to be worked with.
- `CountVectorizer` method (```from sklearn.feature_extraction.text import CountVectorizer```)
  - It tokenizes the string(separates the string into individual words) and gives an integer ID to each token.
  - It counts the occurrence of each of those tokens.
  - The CountVectorizer method automatically converts all tokenized words to their lower case form so that it does not treat words like 'He' and 'he' differently. It does this using the lowercase parameter which is by default set to `True`.
  - It also ignores all punctuation so that words followed by a punctuation mark (for example: 'hello!') are not treated differently than the same words not prefixed or suffixed by a punctuation mark (for example: 'hello'). It does this using the `token_pattern` parameter which has a default regular expression which selects tokens of 2 or more alphanumeric characters.
  - The third parameter to take note of is the `stop_words` parameter. Stop words refer to the most commonly used words in a language. They include words like 'am', 'an', 'and', 'the' etc. By setting this parameter value to english, CountVectorizer will automatically ignore all words(from our input text) that are found in the built in list of english stop words in scikit-learn. This is extremely helpful as stop words can skew our calculations when we are trying to find certain key words that are indicative of spam.

## Support Vector Machine
- SVM Error = Classification Error + Margin Error. Use Gradient Descent to minimize SVM Error.
- The `C` parameter is a constant that attaches itself to the Classification Error.
  
  |`C`|Points Classification|Margin|
  |:---:|:---:|:---:|
  |Large|Good|Small|
  |Small|Some Errors|Large|
- Kernel trick means transforming data into another dimension that has a clear dividing margin between classes of data.
- Linear Kernel (Degree 1: *x*, *y*).
- Polynomial Kernel (Degree 2: *x<sup>2</sup>, xy, y<sup>2</sup>*, Degree 3: *x<sup>3</sup>, x<sup>2</sup>y, xy<sup>2</sup>, y<sup>3</sup>,* etc.).
- RBF (Radio Basis Function) Kernel.
- Gamma (`γ`) parameter:
  - Large Gamma gives tall and pointy mountains → the model tends to overfit.
  - Large Gamma gives short and wide mountains → the model tends to underfit.
- Implementation in sklearn:
  - Hyperparameters:
    - `C`: The `C` parameter.
    - `kernel`: The kernel. The most common ones are 'linear', 'poly', and 'rbf'.
    - `degree`: If the kernel is polynomial, this is the maximum degree of the monomials in the kernel.
    - `gamma` : If the kernel is rbf, this is the Gamma parameter.
  
  ```python
  from sklearn.svm import SVC
  
  # Default
  model = SVC()
  
  # SVC model with a polynomial kernel of degree 4, and a C parameter of 0.1
  model = SVC(kernel='poly', degree=4, C=0.1)
  
  # Fitting the model
  model.fit(x_values, y_values)
  ```
  
## Ensemble Method
- Ensemble method is about combining a bunch of models together to get a better one (Weak Learners → Strong Learners).
- Popular methods:
  - Bagging (Bootstrap aggregating).
  - Boosting (Adaboost).
- Boosting: weight = ln\[accuracy × (1 - accuracy)<sup>-1</sup>\]
- Adaboost in sklearn:
  - Hyperparameters:
    - `base_estimator`: The model utilised for the weak learners.
    - `n_estimators`: The maximum number of weak learners.
    
  ```python
  from sklearn.ensemble import AdaBoostClassifier
  from sklearn.tree import DecisionTreeClassifier
  
  # Default
  model = AdaBoostClassifier()
  
  # Weak learner uses DecisionTree as the model
  model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=4)
  
  # Fitting the model
  model.fit(x_train, y_train)
  
  # Making predictions
  model.predict(x_test)
  ```

## K-means Clustering
- Reference: https://scikit-learn.org/stable/modules/clustering.html#k-means
- K-means algorithm clusters data by separating the dataset into *k* groups of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares. The number of clusters - *k* - needs to be specified before running the algorithm.
  - Step 1: Choose the intial centroids.
  - Loop the next 2 steps until the difference between old and new centroids are less than a threshold.
    - Step 2: Assign each sample to its nearest centroid.
    - Step 3: Create new centroids by averaging the values of all of the samples assigned to each previous centroid.
- Inertia suffers from various drawbacks:
  - Since inertia assumes that clusters are convex and isotropic, it responds poorly to elongated clusters, or manifolds with irregular shapes.
  - Inertia is not a normalised metric. In very high dimensional spaces, Euclidean distances tend to become inflated. Therefore, running a dimensionality reduction algorithm prior to k-means clustering is essential and can help mitigate the problems as well as speed up the computations.
- K-means's hyperparameters:
  - `n_clusters` is the number of clusters.
  - `max_iter` is the maximum number of iterations of k-means algorithm for a single run.
  - `n_init` is the number of time the k-means algorithm will be run with different centroid seeds.
- Implementation in sklearn:
  ```python
  from sklearn.cluster import KMeans
  model = KMeans(n_clusters=8, max_iter=300, n_init=10)
  predictions = model.fit(X)
  ```

## Hierarchical and Density-based Clustering

|Clustering Method|Pros|Cons|
|-|-|-|
|Hierarchical|<ul><li>Resulting hierarchical representation can be very imformative.</li><li>Provides an additional ability to visualise.</li><li>Especially potent when the dataset contains real hierarchical relationships (e.g. Evolutionary biology).</li></ul>|<ul><li>Sensitive to noise and outliers.</li><li>Computational intensive *O(N<sup>2</sup>)*.</li></ul>|
|DBSCAN|<ul><li>No need to specify the number of clusters.</li><li>Flexibility in the shapes & sizes of clusters.</li><li>Able to deal with noise.</li><li>Able to deal with outliers.</li></ul>|<ul><li>Border points that are reachable from two clusters.</li><li>Faces difficulty finding clusters of varying density.</li></ul>|

- Agglomerative Clustering (bottoms-up approach: assume that every point is a cluster and work upwards):
  - Single Link: the linkage method that is more prone to result in elongated shapes that are not necessarily compact or circular.
  - Complete Link.
  - Average Link.
  - Ward: the linkage method that leads to the least increase in variance in the clusters after merging.
- Implementation in sklearn:
  ```python
  from sklearn import cluster, datasets
  
  # Load dataset
  X = datasets.load_iris().data
  
  # Perform clustering
  clust = cluster.AgglomerativeClustering(n_clusters=3, linkage='ward')
  labels = clust.fit_predict(X)  
  ```
- Implmentation in scipy with dendogram:
  ```python
  from scipy.cluster.hierarchy import dendogram, ward, single
  from sklearn import datasets
  import matplotlib.pyplot as plt
  %matplotlib inline
  
  # Load dataset
  X = datasets.load_iris().data[:10]
  
  # Perform Clustering
  linkage_matrix = ward(X)
  
  # Plot dendogram
  dendogram(linkage_matrix)
  
  plt.show()
  ```
- Density-Based Clustering (DBSCAN)
  - Implementation in sklearn
  ```python
  from sklearn import datasets, cluster
  
  # Load dataset
  X = datasets.load_iris().data
  
  # Perform clustering
  db = cluster.DBSCAN(eps=0.5, min_samples=5)
  db.fit(X)
  
  # db.labels_ contains an array representing which cluster each point belongs to. Samples labeled -1 are noise.
  ```

## Gaussian Mixture Models and Clustering Validation
- GMM:
  - Advantages:
    - Soft-clustering (sample membership of multiple clusters).
    - Cluster shape flexibility.
  - Disadvantages:
    - Sensitive to initialisation values.
    - Possible to converge to local optimum.
    - Slow convergence rate.
- Implementation in sklearn:
  ```python
  from sklearn import datasets, mixture
  
  # Load dataset
  X = datasets.load_iris().data[:10]
  
  # GMM
  gmm = mixture.GaussianMixture(n_components=3)
  gmm.fit(X)
  clustering = gmm.predict(X)
  ```
- Cluster Analysis process: Data ↔ Feature Selection/Extraction ↔ Clustering Algorithm Selection & Tuning ↔ Clustering Validation ↔ Results Interpretation ↔ Knowledge.
- Cluster Validation:
  - Categories of cluster validation indices:
    - External Indices.
    - Internal Indices.
    - Relative Indices.
  - Measures:
    - Compactness: How close are the elements to each others?
    - Separability: How distinct are the clusters to each others?
- External Indices: Matching a clustering structure to information we know beforehand.
  
  |Index|Range|Available in sklearn|
  |:-:|:-:|:-:|
  |Adjusted Rand Score|\[-1,1\]|Yes|
  |Fawlks and Mallows|\[0,1\]|Yes|
  |NMI Measure|\[0,1\]|Yes|
  |Jaccard|\[0,1\]|Yes|
  |F-measure|\[0,1\]|Yes|
  |Purity|\[0,1\]|No|
- Internal Indices:
  - Silhouette coefficient (\[-1,1\])

## Feature Scaling
- Feature Scaling is an important step in preprocessing step for some algorithms.
- Feature Scaling Formula: X<sup>'</sup> = (X - X<sub>min</sub>)/(X<sub>max</sub> - X<sub>min</sub>)
- Min/Max Scaling is susceptible to outliers.
- Min/Max Scaler in sklearn:
  ```python
  from sklearn.preprocessing import MinMaxScaler
  import numpy as np
  weights = np.array([[115.],[140.],[175.]])
  scaler = MinMaxScaler()
  rescaled_weights = scaler.fit_transform(weights)
  ```
- Algorithms affected by feature scaling:
  - SVM with 'rbf' `kernel`.
  - KMeans clustering.

## Principal Component Analysis (PCA)
- PCA is a systematised way to transform input features into principal components (PC).
- PCs will be used as new features.
- PCs are directions in data that maximise variance (minimise information loss) when you project/compress down onto them.
- Variance (another definition): technical term in statistics - roughly the "spread" of a data distribution (square of standard deviation).
- The more variance of data along a PC is, the higher that PC is ranked.
- The maximum number of PCs is the number of input features.
- A Scree Plot is a graphical representation of the percentages of variation that each principal component accounts for.
- Dimensionality reduction or dimension reduction is the process of reducing the number of random variables under consideration by obtaining a set of principal variables. It can be divided into feature selection and feature extraction.
- When to use PCA:
  - Identify latent features driving the patterns in data.
  - Dimensionality reduction.
    - Visualise high dimensional data.
    - Reduce noise.
    - Make other algorithms (regression, classification) work better because there are fewer inputs (eigenfaces).
- PCA for Facial Recognition:
  - Pictures of faces generally have high input dimensionality (many pixels).
  - Faces have general patterns that could be captured in smaller number of dimensions (two eyes on top, mouth/chin on bottom, etc.).
- Implementation in sklearn:
  ```python
  from sklearn.decomposition import PCA
  pca = PCA(n_components=2)
  pca.fit(data)
  ```

## Random Projection and Independent Component Analysis (ICA)
- Johnson-Lindenstrauss Lemma: A dataset of *N* points in high-dimensional Euclideanspace can be mapped down to a space in much lower dimension in a way that preserves the distance between the points to a large degree.
- *(1 - esp)*||*u - v*||<sup>2</sup> < ||*p(u) - p(v)*|| < *(1 + esp)*||*u - v*||<sup>2</sup>
- Random Projection is used instead of PCA in case that there are too many dimensions causing PCA's performance to become unacceptable for the situation.
- Random Projection can work either by setting `n_components` or by specifying a value for `eps` and having the algorithm calculate a conservative value for the number of dimensions.
- Implementation in sklearn:
  ```python
  from sklearn import random_projection
  rp = random_projection.SparseRandomProjection()
  new_X = rp.fit_transform(X)
  ```
- Independent Component Analysis (ICA) solves Blind source separation problem.
- FastICA Algorithm:
  1. Centre, whiten X
  2. Choose Initial Random Weight Matrix *W<sub>1</sub>,W<sub>2</sub>, ... , W<sub>n</sub>*
  3. Estimate *W*, containing vectors
  4. Decorrelate *W*
  5. Repeat from step #3 until converged
