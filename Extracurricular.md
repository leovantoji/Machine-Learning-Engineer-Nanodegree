## Tensorflow
- Training a Logistic Classifier: *WX + b = y*.
- Softmax function takes an input vector of *K* real numbers and normalises it into a probability distribution consisting of *K* probabilities.
- Softmax: *S(y<sub>i</sub>) = e<sup>y<sub>i</sub></sup> / Σe<sup>y<sub>i</sub></sup>*
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
  weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
  ```
