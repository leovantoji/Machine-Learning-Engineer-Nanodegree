## Neural Networks
- Error Function tells us how far we are from the solution.
- Log-loss Error Function: A continuous error function allows us to apply gradient descent properly. Log function allows us to avoid negative values.
- Continuous Error Function means Continuous prediction.
- Softmax is a function that provides probabilities for each possible output class.
  ```python
  import numpy as np

  # Write a function that takes as input a list of numbers, and returns
  # the list of values given by the softmax function.
  def softmax(L):
      exp_L = np.exp(L)
      sum_exp = np.sum(exp_L)
      result = [element/sum_exp for element in exp_L]
      return result
  ```
- Maximum Likelihood is the procedure of finding the value of one or more parameters for a given statistic which makes the known likelihood distribution a maxium.
- We want to stay from product because the product of thousands of numbers (all number between 0 and 1) is very small. Additionally, the product could change drastically when one number changes. Thus, we need to use Log function which allows us to change product to sum.
- Cross Entropy loss, or log loss, measures the performance of a classification model whose output is a probability value between 0 and 1. Good model gives low Cross Entropy, while bad model gives high Cross Entropy. 
- *Cross_Entropy = -Σ(y<sub>i</sub>ln(P<sub>i</sub>) + (1-y<sub>i</sub>)ln(1-P<sub>i</sub>))*
  ```python
  import numpy as np

  # Write a function that takes as input two lists Y, P,
  # and returns the float corresponding to their cross-entropy.
  def cross_entropy(Y, P):
      CE = [(-1)*(y*np.log(p) + (1-y)*np.log(1-p)) for y, p in zip(Y, P)]
      return np.sum(CE)
  
  # Solution
  def cross_entropy(Y, P):
      Y = np.float_(Y)
      P = np.float_(P)
      return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
  ```

## Deep Neural Network
- Feedforward is the process neural networks use to turn the input into an output.
- Error function: *E(W) = -m<sup>-1</sup> × Σ(y<sub>i</sub>ln(y_hat<sub>i</sub>) + (1-y<sub>i</sub>)ln(1-y_hat<sub>i</sub>))*
- Backpropagation consists of:
  - Doing a feedforward operation.
  - Comparing the output of the model with the desired output.
  - Calculating the error.
  - Running the feedforward operations backwards (backpropagation) to spread the error to each of the weights.
  - Use this to update the weights, and get a better model.
  - Continue this until we have a model that is good.
- Regularisation is a technique that contrains our optimisation problem to discourage complex models. This technique helps improve generalisation of the model on unseen data.
  - Modify the error function:
    - L1: *E(W) = -m<sup>-1</sup> × Σ(y<sub>i</sub>ln(y_hat<sub>i</sub>) + (1-y<sub>i</sub>)ln(1-y_hat<sub>i</sub>)) + λ(|w<sub>1</sub>| + ... + |w<sub>n</sub>|)*
    - L2: *E(W) = -m<sup>-1</sup> × Σ(y<sub>i</sub>ln(y_hat<sub>i</sub>) + (1-y<sub>i</sub>)ln(1-y_hat<sub>i</sub>)) + λ(w<sub>1</sub><sup>2</sup> + ... + w<sub>n</sub><sup>2</sup>)*
  - L1 vs L2:
    - L1 is used with sparse vector, and it is good for feature selection.
    - L2 is used with non-sparse (small homogeneous weights) vector, and it is generally better for training models.
  - Reguralisation 1: Dropout: Probability each node will be dropped.
    - During training, randomly set some activations to 0. This forces network to not rely on any single node.
    - `tf.keras.layers.Dropout(p=0.5)`
  - Regularisation 2: Early Stopping:
    - Stop training before we have a chance to overfit.
- Exploding Gradients ← Gradient clipping to scale big gradients.
- Vanishing Gradients: Multiplying many small numbers together → Errors due to further back time steps have smaller and smaller gradients → Bias network to capture short-term dependencies.
  - Trick 1: Use different activation functions instead of sigmoid.
    - ReLU (rectified linear unit) prevents f' from shrinking the gradients when x > 0.
    - tanh (hyperbolic tangent function).
  - Trick 2: Initilising weights to identity matrix and biases to zero helps prevent the weights from shrinking to zero.
  - Trick 3: Gated cells (LSTM, GRU, etc.: more complex recurrent unit with gates to control what information is passed through).
- Batch vs. Stochastic Gradient Descent (SGD):
  - SGD typically reaches convergence much faster than Batch Gradient Descent since it updates weights more ferequently.
  - While Batch Gradient Descent computes the gradient using the whole dataset, SGD, also known as incremental gradient descent, tries to find minimums or maximums by iteration from a singly randomly picked training example. Even though theoretically the error is typically noisier than in standard gradient descent, practically it's much better to take a bunch of slightly inaccurate steps than to take one good one.
  - SGD can escape shallow local minima more easily.
  - In order to obtain accurate results with SGD, the data sample should be in a random order, and this is why we want to shuffle the training set for every epoch.
- Learning Rate Decay is to slowly decrease the learning rate over time. A good rule of them is that if the model is not working, decrease the learning rate.
  - *α = (1 + decay_rate * epoch_num)<sup>-1</sup> × α<sub>0</sub>*
- Random restart is used to mitigate local minima/maxima. By performing gradient descent from different random locations, we increase the probability of getting to the global optimum, or at least a pretty good local optimum.
- Gradient Descent with Momentum helps with tackling local minima and is almost always better than traditional Gradient Descent in practice.
  - *V<sub>dW</sub> = 0, V<sub>db</sub> = 0*
  - On iteration *t*, compute *dW* and *db* on the current mini-batch:
    - *V<sub>dW</sub> = βV<sub>dW</sub> + (1-β)dW*.
    - *V<sub>db</sub> = βV<sub>db</sub> + (1-β)db*.
    - *W = W - αV<sub>dW</sub>*
    - *b = b - αV<sub>db</sub>*
    - Hyperparameters: *α* (learning rate), *β* (momentum). *β = 0.9* is the most common value (average or loss of the last 10 gradients).
- Information regarding Keras Optimisers can be found [here](https://keras.io/optimizers/). Some of the most common ones are listed below:
  - SGD: Stochastic Gradient Descent. It uses the following parameters:
    - Learning rate.
    - Momentum (This takes the weighted average of the previous steps, in order to get a bit of momentum and go over bumps, as a way to not get stuck in local minima).
    - Nesterov Momentum (This slows down the gradient when it's close to the solution).
    - `keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)`
  - Adam: Adaptive Moment Estimation uses a more complicated exponential decay that consists of not just considering the average (first moment), but also the variance (second moment) of the previous steps.
    - `keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)`
  - RMSProp: (RMS stands for Root Mean Squared Error) decreases the learning rate by dividing it by an exponentially decaying average of squared gradients. Usually a good choice for Recurrent Neural Networks (RNN).
    - `keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)`
- By removing the final activation function, you can use Neural Network for Regression problems.

## Convolutional Neural Networks
- Deep learning models can be extremely time-consuming to train. In the case the the training is stopped unexpectedly, we will lose lots of work. Therefore, it is necessary to check-point deep learning models during training.
  - Checkpoint Model Improvements:
  ```python
  # Compile model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  # checkpoint
  filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
  # Fit the model
  model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=[checkpoint], verbose=0)
  ```
  - Checkpoint Best Neural Network Model Only:
  ```python
  # Checkpoint the weights for best model on validation accuracy
  # Compile model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  # Checkpoint
  filepath="weights.best.hdf5"
  checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
  # Fit the model
  model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=[checkpoint], verbose=0)
  ```
  - Load a Check-pointed Neural Network Model:
  ```python
  # load weights
  model.load_weights("weights.best.hdf5")
  # Compile model (required to make predictions)
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  print("Created model and loaded weights from file")
  ```
- Grid Search Hyperparameters for Deep Learning Models in Python with Keras:
  - Source: [MachineLearningMastery](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)
  - Keras Models in `scikit-learn`: Keras models can be used in `scikit-learn` by wrapping them with the `KerasClassifier` or `KerasRegressor` class. A function that creates and returns a Keras sequential model must be defined and passed to `build_fn` argument when constructing the `KerasClassifier` class.
  ```python
  def create_model():
    ...
    return model
  
  model = KerasClassifier(build_fn=create_model)
  # model = KerasClassifier(build_fn=create_model, epochs=10)
  # def create_model(dropout_rate=0.0):
  #  ...
  #  return model
  # model = KerasClassifier(build_fn=create_model, dropout_rate=0.2)
  ```
  - Tune Batch Size and Number of Epochs. `n_jobs=-1` allows the process to use all cores on the machine.
  ```python
  # Use scikit-learn to grid search the batch size and epochs
  import numpy
  from sklearn.model_selection import GridSearchCV
  from keras.models import Sequential
  from keras.layers import Dense
  from keras.wrappers.scikit_learn import KerasClassifier

  # Function to create model, required for KerasClassifier
  def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
  
  # Create model
  model = KerasClassifier(build_fn=create_model, verbose=0)
  
  # Define the grid search parameters
  batch_size = [10, 20, 40, 60, 80, 100]
  epochs = [10, 50, 100]
  param_grid = dict(batch_size=batch_size, epochs=epochs)
  grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
  grid_result = grid.fit(X, y)
  
  # Summarize results
  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
  ```
  - Tune the Training Optimisation Algorithm:
  ```python
  # Create model
  model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
  
  # Define the grid search parameters
  optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
  param_grid = dict(optimizer=optimizer)
  grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
  grid_result = grid.fit(X, y)
  
  # Summarize results
  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))  
  ```
  - Tips for Hyperparameter Optimisation:
    - **k-fold Cross Validation**. You can see that the results from the examples in this post show some variance. A default cross-validation of 3 was used, but perhaps k=5 or k=10 would be more stable. Carefully choose your cross validation configuration to ensure your results are stable.
    - **Review the Whole Grid**. Do not just focus on the best result, review the whole grid of results and look for trends to support configuration decisions.
    - **Parallelize**. Use all your cores if you can, neural networks are slow to train and we often want to try a lot of different parameters. Consider spinning up a lot of AWS instances.
    - **Use a Sample of Your Dataset**. Because networks are slow to train, try training them on a smaller sample of your training dataset, just to get an idea of general directions of parameters rather than optimal configurations.
    - **Start with Coarse Grids**. Start with coarse-grained grids and zoom into finer grained grids once you can narrow the scope.
    - **Do not Transfer Results**. Results are generally problem specific. Try to avoid favorite configurations on each new problem that you see. It is unlikely that optimal results you discover on one problem will transfer to your next project. Instead look for broader trends like number of layers or relationships between parameters.
    - **Reproducibility is a Problem**. Although we set the seed for the random number generator in NumPy, the results are not 100% reproducible. There is more to reproducibility when grid searching wrapped Keras models than is presented in this post.
- Multilayer Perceptrons (MLPs):
  - Only use fully connected layers. Lots of parameters are used.
  - Only accept vectors as input. 2-D information (e.g. spatial information, etc.) is thrown away.
- CNNs:
  - Also use sparsely connected layers.
  - Also accept matrices as input.
- Convolutional Layers in Keras:
  
  |Arguments|Compulsory/Optional|Description|
  |:-|:-:|:-|
  |`filters`|C|The number of filters|
  |`kernel_size`|C|Number specifying both the height and width of the (square) convolution window|
  |`strides`|O|The stride of the convolution. Default value is 1|
  |`padding`|O|`valid` or `same`. Default value is `valid`|
  |`activation`|O|Typically `relu`. No default value. Nonetheless, it's highly advisable to apply `relu` to every convolutional layer in the network|
  |`input_shape`|C&O|When the first layer after the input layer is a convolutional layer, `input_shape` must be specified. This is a tuple specifying the height, width, and depth (in that order) of the input. If the first layer after the input layer is not a convolutional layer, `input_shape` should not be included|
  
  ```python
  from keras.layers import Conv2D
  # Example 1
  Conv2D(filters=16, kernel_size=2, strides=2, activation='relu', input_shape=(200, 200, 1))
  # Example 2
  Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')
  # Example 3
  Conv2D(filters=64, kernel_size=(2,2), activation='relu')
  ```
  
  - Number of Parameters in a Convolutional Layer: *`Param #` = `filters` × `kernel_size` × `kernel_size` × `depth_of_the_previous_layer` + `filters`*. `depth_of_the_previous_layer` is the last value in `input_shape` tuple. The formula is as such because:
    - There are *`kernel_size` × `kernel_size` × `depth_of_the_previous_layer`* weights per filter.
    - There is one bias term per filter, the convolutional layer has *`filters`* biases.
  - Shape of a Convolutional Layer depends on the supplied values of `kernel_size`, `input_shape`, `padding`, and `stride`:
    
    |`padding`|height|width|
    |:-:|:-:|:-:|
    |'same'|ceil(float(`input_shape[0]`)/float(`stride`)|ceil(float(`input_shape[1]`)/float(`stride`)|
    |'valid'|ceil(float(`input_shape[0]` - `kernel_size` + 1)/float(`stride`)|ceil(float(`input_shape[1]` - `kernel_size` + 1)/float(`stride`)|
    
- Max Pooling Layers are used to reduce the dimensionality of the previous convolutional layers. Max Pooling Layers in Keras:

  |Arguments|Compulsory/Optional|Description|
  |:-|:-:|:-|
  |`pool_size`|C|Number specifying the height and width of the pooling window|
  |`strides`|O|The vertical and horizontal stride. Default value is `pool_size`|
  |`padding`|O|`valid` or `same`. Default value is `valid`|
  
  ```python
  from keras.layers import MaxPooling2D
  MaxPooling2D(pool_size=2, strides=2)
  ```
- CNNs for Image Classification:
  - Create a `Sequential` model:
  ```python
  from keras.models import Sequential
  ```
  - Import necessary layers:
  ```python
  from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
  ```
  - Add layers to the network:
  ```python
  model = Sequential()
  model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(32, 32, 3)))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Flatten())
  model.add(Dense(500, activation='relu'))
  model.add(Dense(10, activation='softmax'))
  ```
  - Things to remember:
    - Always add a ReLU activation function to the `Conv2D` layers in your CNN. With the exception of the final layer in the network, `Dense` layers should also have a ReLU activation function.
    - When constructing a network for classification, the final layer in the network should be a `Dense` layer with a softmax activation function. The number of nodes in the final layer should equal the total number of classes in the dataset.
    - [Andrej Karpathy's tumblr](https://lossfunctions.tumblr.com/)
