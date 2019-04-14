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
