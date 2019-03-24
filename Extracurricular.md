## Tensorflow
- `tf.placeholder()` and `tf.constant()` can't be modified.
- `tf.Variable()` class creates a tensor with an initial value that can be modified.
- `tf.global_variables_initializer()` is used to initialise the state of all the Variable tensors.
  ```python
  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init)
  ```
- Training a Logistic Classifier: *WX + b = y*.
- Softmax function takes an input vector of *K* real numbers and normalises it into a probability distribution consisting of *K* probabilities.
- Softmax: *S(y<sub>i</sub>) = e<sup>y<sub>i</sub></sup> / Σe<sup>y<sub>i</sub></sup>
