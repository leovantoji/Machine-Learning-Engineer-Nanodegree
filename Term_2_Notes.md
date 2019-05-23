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
  - We want the algorithm to learn the invariant representation of the image. Types of invariance:
    - Scale invariance: size of the object.
    - Rotation invariance: angle of the object.
    - Translation invariance: position of the object (left, centre, right).
  - Data augmentation helps prevent overfitting. Image augmentation in Keras:
    ```python
    from keras.preprocessing.image import ImageDataGenerator
    
    # create and configure augmented image generator
    datagen = ImageDataGenerator(
      width_shift_range=0.1, # randomly shift images horizontally (10% of total width)
      height_shift_range=0.1, # randomly shift images vertically (10% of total height)
      horizontal_flip=True) # randomly flipp images horizontally
    
    # fit augmented image generator on data
    datagen.fit(X_train)
    
    # visualise original and augmented images
    import matplotlib.pyplot as plt
    
    # take subset of training data
    X_train_subset = X_train[:12] # first 12 images
    
    # visualise subset of training data
    fig = plt.figure(figsize=(20,2))
    for i in range(0, len(X_train_subset)):
      ax = fig.add_subplot(1, 12, i+1)
      ax.imshow(X_train_subset[i])
    fig.suptitle("Subset of Original Training Images", fontsize=20)
    
    # visualise augmented images
    fig = plt.figure(figsize(20,2))
    for X_batch in datagen.flow(X_train_subset, batch_size=12):
      for i in range(0, 12):
        ax = fig.add_subplot(1, 12, i+1)
        ax.imshow(X_batch[i])
      fig.suptitle("Augmented Images", fontsize=20)
      break;
      
    # train the model
    from keras.callbacks import ModelCheckpoint
    
    batch_size = 32
    epochs = 100
    
    checkpointer = ModelCheckpoint(filepath="aug_model.weights.best.hdf5", verbose=0, save_best_only=True)
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                          steps_per_epoch=X_train.shape[0] // batch_size,
                          epochs=epochs, verbose=0, callbacks=[checkpointer],
                          validation_data=(X_valid, y_valid))
    ```
  - Things to remember:
    - The filter is often a square, and `kernel_size` is usually 2x2 at the smallest to 5x5 at the largest. The `strides` is usually set to 1, which is the default value in keras. As for `padding`, it is believed that 'same' would yield better results than 'valid'. This choice of hyperparameters makes the width and height of the convolutional layer the same as the previous layer. The number of `filters` often slowly increase in sequence.
    - Common setting for Max Pooling layer is that `pool_size` and `strides` are both 2.
    - Always add a ReLU activation function to the `Conv2D` layers in your CNN. With the exception of the final layer in the network, `Dense` layers should also have a ReLU activation function.
    - When constructing a network for classification, the final layer in the network should be a `Dense` layer with a softmax activation function. The number of nodes in the final layer should equal the total number of classes in the dataset.
    - [Andrej Karpathy's tumblr](https://lossfunctions.tumblr.com/)
- Transfer Learning involves taking a pre-trained neural network and adapating the neural network to a new, different data set.
  - 4 main scenarios of Transfer Learning.
    
    |Case|New Data|Similarity to Original Data|What to do?|
    |:-:|:-:|:-:|:-:|
    |1|Small|Similar|End of ConvNet|
    |2|Small|Different|Start of ConvNet|
    |3|Large|Similar|Fine-tune|
    |4|Large|Different|Fine-tune or Retrain|
  - Case 1: Small data set with similar data.
    - Slice off the end of the neural network.
    - Add a new fully connected layer that matches the number of classes in the new data set.
    - Randomise the weights of the new fully connected layer; freeze all the weights from the pre-trained network.
    - Train the network to update the weights of the new fully connected layer.
    - To avoid overfitting on the small data set, the weights of the original network will be held constant rather than re-training the weights.
    - Since the data set are similar, images from each data set will have similar higher level features. Therefore, most or all of the pre-trained neural network layers already contain relevant information about the new data set and should be kept.
  - Case 2: Small data set with different data.
    - Slice off most of the pre-trained layers near the beginning of the network.
    - Add to the remaining pre-trained layers a new fully connected layer that matches the number of classes in the new data set.
    - Randomise the weights of the new fully connected layer; freeze all the weights from the pre-trained network.
    - Train the network to update the weights of the new fully connected layer.
    - To combat overfitting, the weights of the original neural network will be held constant.
    - Since the 2 data sets are different, the new network will only use the layers containing lower level features.
  - Case 3: Large data set with similar data.
    - Remove the last fully connected layer and replace with a layer matching the number of classes in the new data set.
    - Randomly initialise the weights in the new fully connected layer.
    - Initialise the rest of the weights using the pre-trained weights.
    - Re-train the entire neural network.
    - Since overfitting is not much of a concern when training on a large data set, you can re-train all of the weights.
    - The entire neural network is used because the original training set and the new data set share higher level features.
  - Case 4: Large data set with different data.
    - Remove the last fully connected layer and replace with a layer matching the number of classes in the new data set.
    - Re-train the network from scratch with randomly initialised weights.
    - Alternatively, you could just use the same strategy as that of Case 3.
    - Even though the data set is different from the training data, initializing the weights from the pre-trained network might make training faster. So this case is exactly the same as the case with a large, similar data set. If using the pre-trained network as a starting point does not produce a successful model, another option is to randomly initialize the convolutional neural network weights and train the network from scratch.

## Reinforcement Learning
- Reinforcement Learning Key Concepts:
  - **Agent**: takes actions.
  - **Environment**: the world in which the agent exists and operates.
  - **Action**: a move the agent can make in the environment.
  - **Observation**: of the environment after taking actions.
  - **State**: a situation which the agent perceives.
  - **Reward**: feedback that measures the success or failure of the agent's action.
  - **Total Reward**: *G<sub>t</sub> = Σr<sub>i</sub>*. 
  - Because *G<sub>t</sub>* can go to infinity, there has to be a **discount factor** which places more weight on short-term timestamp reward and less weight on long-term timestamp from the current state reward. Thus, we have *G<sub>t</sub> = Σγ<sup>i</sup>r<sub>i</sub>*.
  - **Q-function** captures the **expected total future reward** an agent in state, *s*, can receive by executing a certain action, *a*. Thus, we have *Q(s,a) = E[G<sub>t</sub>]*
  - The agent needs a **policy π(s)**, to infer the **best action to take** at its state, *s*.
  - **Strategy**: the policy should choose an action that maximises future reward. *π\*(s) = argmax Q(s,a)*.
- A **task** is an instance of the reinforcement learning (RL) problem.
  - **Episodic tasks** are tasks with well-defined ending points. Interaction ends at some time step *T*: *S<sub>0</sub>, A<sub>0</sub>, R<sub>1</sub>, S<sub>1</sub>, A<sub>1</sub>, ..., R<sub>T</sub>, S<sub>T</sub>*. This whole sequence of interaction (State, Action, Reward), from start to finish, is called an episode. Episodic tasks come to an end whenever the agent reaches a **terminal state**.
  - **Continuing tasks** are tasks that continue forever, without end.
- The task suffers the problem of **sparse reward** when the reward signal is largely uninformative (reward only comes at the end of the episode).
- **Reward Hypothesis**: All goals can be framed as the maximisation of expected cumulative reward.
- **Markov Decision Process (MDP)** is a discrete time stochastic control process. It provides a mathematical framework for modeling decision making in situations where outcomes are partly random and partly under the control of a decision maker.
- A (finite) MDP is defined by:
  - a (finite) set of states *S* - known to the agent.
  - a (finite) set of actions *A* - known to the agent.
  - a (finite) set of rewards *R* - not known to the agent.
  - the one-step dynamics of the environment - not known to the agent.
    - *p(s<sup>'</sup>,r|s,a) = P(S<sub>t+1</sub>=s<sup>'</sup>, R<sub>t+1</sub>=r|S<sub>t</sub>=s, A<sub>t</sub>=a)* for all *s, s<sup>'</sup>, a, and r*.
  - a discount rate *γ∈\[0,1\]* - known to the agent.
- Types of policy:
  - A Deterministic policy is a mapping *π: S → A*.
  - A Stochastic policy is a mapping *π: S x A → \[0,1\]*
    - *π(a|s) = P(A<sub>t</sub>=a|S<sub>t</sub>=s)*
- **Bellman Expectation Equation** (for *v<sub>π</sub>*) expressesthe value of any state *s* in terms of the *expected* immediate reward and the *expected* value of the next state:
  - *v<sub>π</sub>(s) = E<sub>π</sub>\[R<sub>t+1</sub> + γv<sub>π</sub>(S<sub>t+1</sub>)|S<sub>t</sub>=s\]*.
  - In the event that the agent's policy *π* is deterministic, the agent selects action *π(s)* when in state *s*, and the Bellman Expectation Equation can be rewritten as the sum over two variables (*s<sup>'</sup>* and *r*): *v<sub>π</sub>(s) = **∑** p(s<sup>'</sup>,r|s,π(s))(r + γv<sub>π</sub>(s<sup>'</sup>)*.
  - If the agent's policy *π* is stochastic, the agent selects action *a* with probability *π(a∣s)* when in state *s*, and the Bellman Expectation Equation can be rewritten as the sum over three variables (*s<sup>'</sup>*, *r*, and *a*): *v<sub>π</sub>(s) = **∑** π(a|s)p(s<sup>'</sup>,r|s,a)(r + γv<sub>π</sub>(s<sup>'</sup>)*.
- A policy *π<sup>'</sup>* is better than another policy *π*, if and only if its value function is greater than that of the other policy for all state: *π<sup>'</sup> ≥ π* if and only if *v<sub>π<sup>'</sup></sub>(s) ≥ v<sub>π</sub>(s)* for all *s ∈ S*.
- An optimal policy *π<sub>\*</sub>* satisfies *π<sub>\*</sub> ≥ π* for all *π*. An optimal policy is guaranteed to exist, but may not be unique. The optimal state-value function is denoted as *v<sub>\*</sub>*. The optimal action-value function is *q<sub>\*</sub>*.
- State-value function vs. Action-value function for policy *π*:
  - The value of state *s* under a policy *π* is calculated as *v<sub>π</sub>(s) = E<sub>π</sub>\[G<sub>t</sub>\|S<sub>t</sub>=s\]*. For each state *s*, it yields the expected return if the agent starts in state *s* and then uses the policy to choose its action for all time steps.
  - The value of taking action *a* in state *s* under a policy *π* is calculated as *q<sub>π</sub>(s,a) = E<sub>π</sub>\[G<sub>t</sub>\|S<sub>t</sub>=s, A<sub>t</sub>=a\]*. For each state *s* and action *a*, it yields the expected return if the agent starts in state *s* then chooses action *a* and then uses the policy to choose its actions for all time steps.
- Once the agent determines the optimal action-value function *q<sub>\*</sub>*, it can quickly obtain an optimal policy *π* by setting *π<sub>\*</sub>(s) =* argmax *q<sub>\*</sub>(s,a)*.
- Deep Reinforcement Learning Algorithms:
  - **Value learning**: Find *Q(s,a). a = argmax Q(s,a)*.
  - **Policy learning**: Find *π(s)*. Sample *a ~ π(s)*.
- **Deep Q Networks (DQN)**:
  
  |Input|NN|Output|
  |:-:|:-:|:-:|
  |<ul><li>state, *s*</li><li>action, *a*</li></ul>|Deep NN|*Q(s,a)*|
  |state, *s*|Deep NN|<ul><li>*Q(s,a<sub>1</sub>)*</li><li>*Q(s,a<sub>2</sub>)*</li></ul>|
- Downsides of Q-learning:
  - Complexity: 
    - Can model scenarios where the action space is discrete and small.
    - Cannot handle continuous action spaces.
  - Flexibility:
    - Cannot learn stochastic policies since policy is deterministically computed from the Q function.
- **Policy Gradient**: directly optimises the policy, while DQN tries to approximate Q and infer the optimal policy.
  - Run a policy for a while.
  - Increase probability of actions that lead to high rewards.
  - Decrease the probability of actions that lead to low/no rewards.

## Dynamic Programming
- In the **dynamic programming** setting, the agent has full knowledge of the MDP. (This is much easier than reinforcement learning setting, where the agent initially knows nothing about how the environment decides state and reward and must learn entirely from interaction how to select actions.)
- **Iterative Policy Evaluation** is an algorithm used in the dynamic programming setting to estimate the state-value function *v<sub>π</sub>* corresponding to a policy *π*.
- **Estimation of Action Values**: In the dynamic programming setting, it is possible to quickly obtain the action-value function *q<sub>π</sub>* from the state-value function *v<sub>π</sub>* with the equation: *q<sub>π</sub>(s,a) = **∑** p(s<sup>'</sup>,r|s,a)(r + γv<sub>π</sub>(s<sup>'</sup>)*
- **Policy Improvement**: An improved policy *π<sup>'</sup>* is a policy that satifies for each state *s ∈ S* and *a ∈ A(s)*: *π<sup>'</sup>(a|s) = 0* if *a ∉ argmax Q(s, a<sup>'</sup>)*. In other words, any policy that (for each state) assigns zero probability to the actions that do not maximise the action-value function estimate (for that state) is an improved policy.
- **Policy Iteration** proceeds as a series of alternating policy evaluation and improvement steps. Policy iteration is guaranteed to find the optimal policy for any finite MDP in a finite number of iterations.
- **Truncated Policy Evaluation** only performs a fixed number of sweeps through the state space.
- **Value Iteration**: each sweep over the state space *S* effectively performs both policy evaluation and policy improvement. Value iteration is guaranteed to find the optimal policy *π<sub>\*</sub>* for any finite MDP.

## Monte Carlo Methods
- The **Prediction Problem**: Given a policy *π*, determine *v<sub>π</sub>* or (*q<sub>π</sub>*).
- The **Off-Policy Method** for the Prediction Problem: Generate episodes from following policy *b*, where *b ≠ π*. The generated episodes will then be used to estimate *v<sub>π</sub>*.
- The **On-Policy Method** for the Prediction Problem: Generate episodes from following policy *π*. The generated episodes will then be used to estimate *v<sub>π</sub>*.
- Each occurence of state *s ∈ S* in an episode is called a **visit** to *s*.
  - First-visit MC method is unbiased.
  - Every-visit MC method is biased. Initially, every-visit MC has lower mean squared error (MSE), but as more episodes are collected, first-visit MC attains better MSE.
  - Both first-visit and every-visit MC method are guaranteed to converge to the true value function, as the number of visits to each state approaches infinity.
- We won't use MC prediction to estimate the action-values corresponding to a deterministic policy; this is because many state-action pairs will never be visited (since a deterministic policy always chooses the same action from each state). Instead, so that convergence is guaranteed, we will only estimate action-value functions corresponding to policies where each action has a nonzero probability of being selected from each state.
- The **Control Problem**: How might an agent determine an optimal policy *π<sub>\*</sub>* from interactions with the environment?
- **Generalised Policy Iteration**:
  - Initialise *N(s,a) = 0* for all *s ∈ S, a ∈ A(s)*.
  - Initialise *Q(s,a) = 0* for all *s ∈ S, a ∈ A(s)*.
  - Begin with starting policy *π*.
  - Repeat:
    - Policy Evaluation:
      - Generate an episode *S<sub>0</sub>, A<sub>0</sub>, R<sub>1</sub>, ..., S<sub>r</sub>* using *π*.
      - Use the episode to update *Q*: *Q(S<sub>t</sub>, A<sub>t</sub>) += 1/N(S<sub>t</sub>, A<sub>t</sub>) × (G<sub>t</sub> - Q(S<sub>t</sub>, A<sub>t</sub>))*
    - Policy Improvement:
      - Greedy Policy: Use *Q* to improve *π*: *π(s) ← argmax Q(s,a)* for all *s ∈ S, a ∈ A(s)*.
      - Epsilon-Greedy Policy: *ε ∈ \[0,1\]*. Agent selects greedy action with *(1 - ε + ε/|A(s)|)* probability and randomly selects an action with *ε/|A(s)|* probability.
- Notes on value of *ε*:
  - *ε = 0* yields an epsilon-greedy policy that is the same as greedy policy.
  - No value of *ε* can yield an epsilon-greedy policy that is guaranteed to always select a non-greedy action.
  - *ε = 1* yields an epsilon-greedy policy that is equivalent to the equiprobable random policy.
  - *ε > 0* yields an epsilon-greedy policy that is guaranteed to sometimes select a greedy and sometimes select a non-greedy action.
- One potential solution to the Exploration-Exploitation Dilemma is implemented by gradually modifying the value of *ε* when constructing epsilon-greedy policies.
- Greedy in the Limit with Infinite Exploration (GLIE): If every state-action pair *(s,a)* is visited infinitely many times, and the policy converges to a policy that is greedy with respect to the action-value function estimate *Q*, then MC Control is guaranteed to converge to the optimal policy (in the limit as the algorithm is run for infinitely many episodes). These condition ensures that the agent continues to explore for all time steps, and the agent gradually exploits more (explores less). To satisfy this, set *ε<sub>i</sub> = 1/i* for all time step *i*.
