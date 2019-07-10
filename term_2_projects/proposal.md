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
_(approx. 2-3 paragraphs)_

In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem.

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
  - Fine-tune InceptionV3 Model.
    - Load base InceptionV3 model with pre-loaded "imagenet" weights.
    - Freeze the first 254 layers.
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
    
_(approx. 1 page)_

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.
