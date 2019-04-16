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
