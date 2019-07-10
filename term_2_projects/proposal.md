# Machine Learning Engineer Nanodegree
## Capstone Proposal
**Car Make and Model Image Recognition**
- Chu Nguyen Van
- July 10, 2019

### Domain Background

_(approx. 1-2 paragraphs)_

In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required.

### Problem Statement
_(approx. 1 paragraph)_

In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once).

### Datasets and Inputs
#### Dataset Description
- The Cars dataset contains 16,185 images of 196 classes of cars. 
- The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split.
- Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe.

#### Acknowledgement
- Data source and banner image: http://ai.stanford.edu/~jkrause/cars/car_dataset.html contains all bounding boxes and labels for both training and tests.
- **3D Object Representations for Fine-Grained Categorization**
  - Jonathan Krause, Michael Stark, Jia Deng, Li Fei-Fei
  - 4th IEEE Workshop on 3D Representation and Recognition, at ICCV 2013 (3dRR-13). Sydney, Australia. Dec. 8, 2013.

### Solution Statement
- Inspired by the results of the Deep Learning research for Cancer Detection by Sebastian Thrun, I intend to use Transfer Learning to solve this classification problem. To be specific, I will use the Xception model initialised with "imagenet" weights. Since the cars dataset is much smaller than the ImageNet dataset, retraining the whole model will probably result in overfitting. Thus, it is my intention to keep the majority of the pre-loaded weights intact so that I can leverage the top model to extract higher level features of the image. The last few layers of the top model will be retrained and subsequently connected with my own logistic layers specific to this problem. 

### Benchmark Model
- A random guess with equal probability assigned for each car class has a ~0.5% (= 1/196) probability of being accurate.
- I will train a simple CNN model from scratch to serve as a benchmark for my transfer learning model.
- The architecture of the benchmarking model is shown below.
  ```python
  model = Sequential()
  model.add(Conv2D(filters=16, kernel_size=2, padding="same", activation="relu", input_shape=train_tensors[0].shape))
  model.add(Conv2D(filters=16, kernel_size=2, padding="same", activation="relu"))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))
  model.add(Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))
  model.add(MaxPooling2D(pool_size=2))
  model.add(Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"))
  model.add(Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"))
  model.add(GlobalAveragePooling2D())
  model.add(Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
  model.add(Dropout(rate=0.5))
  model.add(Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
  model.add(Dropout(rate=0.5))  
  model.add(Dense(N_CLASSES, activation='softmax')) # N_CLASSES = 196
  ```

### Evaluation Metrics
- An accuracy score comparing the model predictions of test images against the true test label will be computed. The higher the accuracy score is, the better the model is.
- A confusion matrix can also be used to show the car classes that the model struggles the most to discern.

### Project Design
- **Programming language**: Python 3.7
- **Library/Framework**: Pandas, Numpy, Scikit-learn, Tensorflow Keras
- **Workflow**:
  - Data Collection: the cars dataset is originated from Stanford University AI Lab and was downloaded from the aforementioned website in the *Datasets and Inputs* section
  - Explore the dataset: construct a bar chart to understand the number of car images associated with each car make and model.
  - Image Preprocessing: crop the car out of the original image based on the provided bounding box information. This will eliminate all noises in the image so that the model can have better accuracy thanks to cleaner input.
  - Data preparation: 
    - Split original training dataset into 3 sets:
      - Training: 6515 images – 80% of the original training set.
      - Validation: 815 images – 10% of the original training set.
      - Private Test: 814 images – 10% of the original training set.
    - Use the original test set as a Public Test set of 8041 images.
    - Perform one-hot encoding on the car labels.
    - Transform datasets into 4D tensors so that they can be used as input for Tensorflow Keras CNN.
    - Perform image augmentation to add more variety to the images.
  - Train benchmarking model.
  - Test benchmarking model against Private and Public Test set.
  - Validate benchmarking model performance by charting out the learning curve.
  - Fine-tune Xception Model.
    - Load base Xception model with pre-loaded "imagenet" weights.
    - Freeze the first 94 layers.
    - Make the rest of the layers trainable.
    - Append a Batch Normalization layer, a Global Average Pooling 2D layer, 2 fully connected layers with kernel regularization and a Dense layer with softmax activation function to the top model.
    - Train the final model with Model Checkpoint, Early Stopping if there is no improvement in "val_acc" and Reduce Learning Rate if there is no improvement in "val_loss".
  - Test transfer learning model against Private and Public Test set.
  - Validate transfer learning model performance by charting out the learning curve.
  - Compare the accuracy of the transfer learning model against the benchmarking model.

### Reference
- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
