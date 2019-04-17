# Udacity Machine Learning Engineer Nanodegree

## Overview
|Term|Official Duration|Date of Compeletion|Projects|
|:-:|:-:|:-:|-|
|1|20 Feb - 8 May 2019|20 Mar 2019|<ul><li>Predicting Boston Housing Prices</li><li>Finding Donors for CharityML</li><li>Creating Customer Segments</li></ul>|
|2|10 Apr - 18 Aug 2019||<ul><li>Dog Breed Classifier</li><li>Teach a Quadcopter how to fly</li><li>Capstone Project</li></ul>|

## Projects
### 1. Predicting Boston Housing Prices:
- **Machine Learning Concepts:** Regression.
- **Project Overview:** The Boston housing market is highly competitive, and you want to be the best real estate agent in the area. To compete with your peers, you decide to leverage a few basic machine learning concepts to assist you and a client with finding the best selling price for their home. Luckily, you’ve come across the Boston Housing dataset which contains aggregated data on various features for houses in Greater Boston communities, including the median value of homes for each of those areas. Your task is to build an optimal model based on a statistical analysis with the tools available. This model will then be used to estimate the best selling price for your clients' homes.
- **Project Highlights:** 
  - How to explore data and observe features.
  - How to train and test models.
  - How to identify potential problems, such as errors due to bias or variance.
  - How to apply techniques to improve the model, such as cross-validation and grid search.

### 2. Finding Donors for CharityML:
- **Machine Learning Concepts:** Decision Tree, Random Forest, Support Vector Machine.
- **Project Overview:** In this project, you will employ several supervised algorithms of your choice to accurately model individuals' income using data collected from the 1994 U.S. Census. You will then choose the best candidate algorithm from preliminary results and further optimize this algorithm to best model the data. Your goal with this implementation is to construct a model that accurately predicts whether an individual makes more than $50,000. This sort of task can arise in a non-profit setting, where organizations survive on donations. Understanding an individual's income can help a non-profit better understand how large of a donation to request, or whether or not they should reach out to begin with. While it can be difficult to determine an individual's general income bracket directly from public sources, we can (as we will see) infer this value from other publically available features. The dataset for this project originates from the UCI Machine Learning Repository. The datset was donated by Ron Kohavi and Barry Becker, after being published in the article "Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid". You can find the article by Ron Kohavi online. The data we investigate here consists of small changes to the original dataset, such as removing the `fnlwgt` feature and records with missing or ill-formatted entries.
- **Project Highlights:** 
  - How to identify when preprocessing is needed, and how to apply it.
  - How to establish a benchmark for a solution to the problem.
  - What each of several supervised learning algorithms accomplishes given a specific dataset.
  - How to investigate whether a candidate solution model is adequate for the problem.

### 3. Creating Customer Segments:
- **Machine Learning Concepts:** K-Means Clustering, Gaussian Mixture Model.
- **Project Overview:** In this project you will apply unsupervised learning techniques on product spending data collected for customers of a wholesale distributor in Lisbon, Portugal to identify customer segments hidden in the data. You will first explore the data by selecting a small subset to sample and determine if any product categories highly correlate with one another. Afterwards, you will preprocess the data by scaling each product category and then identifying (and removing) unwanted outliers. With the good, clean customer spending data, you will apply PCA transformations to the data and implement clustering algorithms to segment the transformed customer data. Finally, you will compare the segmentation found with an additional labeling and consider ways this information could assist the wholesale distributor with future service changes.
- **Project Highlights:**
  - How to apply preprocessing techniques such as feature scaling and outlier detection.
  - How to interpret data points that have been scaled, transformed, or reduced from PCA.
  - How to analyze PCA dimensions and construct a new feature space.
  - How to optimally cluster a set of data to find hidden patterns in a dataset.
  - How to assess information given by cluster data and use it in a meaningful way.

### 4. Dog Breed Classifier:
- **Machine Learning Concepts:** Convolutional Neural Networks (CNN).
- **Project Overview:** Welcome to the Convolutional Neural Networks (CNN) project! In this project, you will learn how to build a pipeline to process real-world, user-supplied images. Given an image of a dog, your algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed. Along with exploring state-of-the-art CNN models for classification, you will make important design decisions about the user experience for your app. Our goal is that by completing this lab, you understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline. Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer. Your imperfect solution will nonetheless create a fun user experience!

### 5. Teaching a Quadcopter how to fly:
- **Machine Learning Concepts:** Deep Reinforcement Learning.
- **Project Overview:** The Quadcopter or Quadrotor Helicopter is becoming an increasingly popular aircraft for both personal and professional use. Its maneuverability lends itself to many applications, from last-mile delivery to cinematography, from acrobatics to search-and-rescue. Most quadcopters have 4 motors to provide thrust, although some other models with 6 or 8 motors are also sometimes referred to as quadcopters. Multiple points of thrust with the center of gravity in the middle improves stability and enables a variety of flying behaviors. But it also comes at a price–the high complexity of controlling such an aircraft makes it almost impossible to manually control each individual motor's thrust. So, most commercial quadcopters try to simplify the flying controls by accepting a single thrust magnitude and yaw/pitch/roll controls, making it much more intuitive and fun. The next step in this evolution is to enable quadcopters to autonomously achieve desired control behaviors such as takeoff and landing. You could design these controls with a classic approach (say, by implementing PID controllers). Or, you can use reinforcement learning to build agents that can learn these behaviors on their own. This is what you are going to do in this project!
- **Project Highlights:** In this project, you will design your own reinforcement learning task and an agent to complete it. Note that getting a reinforcement learning agent to learn what you actually want it to learn can be hard, and very time consuming. For this project, we strongly encourage you to take the time to tweak your task and agent until your agent is able to demonstrate that it has learned your chosen task, but this is not necessary to complete the project. As long as you take the time to describe many attempts at specifying a reasonable reward function and a well-designed agent with well-informed hyperparameters, this is enough to pass the project.

## Course Contents:
- Gradient Descent
- Linear Regression
- Logistic Regression
- Naive Bayes
- Decision Trees
- Perceptron Algorithm
- Support Vector Machines
- Ensemble Methods
- Clustering
- Hierarchical and Density-based Clustering
- Gaussian Mixture Models and Cluster Validation
- Feature Scaling
- PCA
- Random Projection and ICA
