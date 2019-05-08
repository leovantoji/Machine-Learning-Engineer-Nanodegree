## Tensorflow
- Training a Logistic Classifier: *WX + b = y*.
- Softmax function takes an input vector of *K* real numbers and normalises it into a probability distribution consisting of *K* probabilities.
- Softmax: *S(y<sub>i</sub>) = e<sup>y<sub>i</sub></sup> / Σe<sup>y<sub>i</sub></sup>*
  ```python
  x = tf.nn.softmax([2.0, 1.0, .2])
  ```
- `tf.placeholder()` and `tf.constant()` can't be modified.
- `tf.Variable()` class creates a tensor with an initial value that can be modified.
- `tf.global_variables_initializer()` is used to initialise the state of all the Variable tensors.
  ```python
  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init)
  ```
- Initialising the weights with random numbers from a normal distribution is good practice. Randomising the weights helps the model from becoming stuck in the same place every time you train it.
- `tf.truncated_normal` function generates random numbers from a normal distribution.
  ```python
  n_features = 120
  n_labels = 5
  weights = tf.Variable(tf.truncated_normal(shape=(n_features, n_labels)))
  bias = tf.Variable(tf.zeros(n_labels))
  ```
- Rectified Linear Units (ReLU): *f(x) = max(x,0)*. ReLU is used for the hidden layers instead of sigmoid function. The reason is that the derivative of the sigmoid maxes out at 0.25. This means when you're performing backpropagation with sigmoid units, the errors going back into the network will be shrunk by at least 75% at every layer. For layers close to the input layer, the weight updates will be tiny if you have a lot of layers and those weights will take a really long time to train.
- Cross entropy in tensorflow.
  ```python
  x = tf.reduce_sum([1, 2, 3, 4, 5])
  ```
