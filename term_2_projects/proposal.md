# Machine Learning Engineer Nanodegree
## Capstone Proposal
Chu Nguyen Van
July 10, 2019

## Proposal
_(approx. 2-3 pages)_

### Domain Background
_(approx. 1-2 paragraphs)_

In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required.

### Problem Statement
_(approx. 1 paragraph)_

In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once).

### Datasets and Inputs
#### Dataset Description
The Cars dataset contains 16,185 images of 196 classes of cars. The data is split into 8,144 training images and 8,041 testing images, where each class has been split roughly in a 50-50 split. Classes are typically at the level of Make, Model, Year, e.g. 2012 Tesla Model S or 2012 BMW M3 coupe.

#### Acknowledgement
Data source and banner image: http://ai.stanford.edu/~jkrause/cars/car_dataset.html contains all bounding boxes and labels for both training and tests.\
\
**3D Object Representations for Fine-Grained Categorization**\
Jonathan Krause, Michael Stark, Jia Deng, Li Fei-Fei\
4th IEEE Workshop on 3D Representation and Recognition, at ICCV 2013 (3dRR-13). Sydney, Australia. Dec. 8, 2013.

### Solution Statement
_(approx. 1 paragraph)_

In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once).

### Benchmark Model
_(approximately 1-2 paragraphs)_

In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail.

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms).

### Project Design
Programming language: Python 3.7
Library/Framework: Pandas, Numpy, Scikit-learn, Tensorflow Keras
Workflow:
- Data Collection: the cars dataset is originated from Stanford University AI Lab and was downloaded from the aforementioned website in the Datasets and Inputs section
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
  - Append the following model to the base model.
  ```python
  model = Sequential()
  model.add(Xception_model)
  model.add(BatchNormalization())
  model.add(GlobalAveragePooling2D())
  model.add(Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
  model.add(Dropout(rate=0.5))
  model.add(Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01)))
  model.add(Dropout(rate=0.5))
  model.add(Dense(N_CLASSES, activation="softmax")) # N_CLASSES = 196
  ```
  - Train the combined model with Model Checkpoint, Early Stopping if there is no improvement in "val_acc" and Reduce Learning Rate if there is no improvement in "val_loss".
- Test transfer learning model against Private and Public Test set.
- Validate transfer learning model performance by charting out the learning curve.
- Compare the accuracy of the transfer learning model against the benchmarking model.

## Reference
- [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)
