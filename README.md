# Sentiment Analysis Comparative Analysis

This project aims to perform sentiment analysis on a Twitter dataset using different models and techniques. The goal is to compare the performance of various models and evaluate their effectiveness in predicting sentiment.

## Dataset

The dataset used in this project is named "training.1600000.processed.noemoticon.csv". It consists of a collection of tweets and their corresponding sentiment labels. The dataset has been preprocessed, including the removal of mentions and hashtags, and the sentiment labels have been converted to binary format (0 for negative sentiment and 1 for positive sentiment).

## Models and Techniques

The following models and techniques have been applied in the comparative analysis:

- Naive Bayes: A probabilistic classifier that applies Bayes' theorem with strong independence assumptions between the features.
- Support Vector Machines (SVM): A supervised machine learning algorithm that classifies data by finding the hyperplane that maximally separates the classes.
- Bidirectional LSTM with Word2Vec: A deep learning model that utilizes bidirectional long short-term memory (LSTM) layers to capture contextual information from text, combined with word embeddings generated using Word2Vec.
- Transformer-based model with BERT-based word embedding: A transformer-based model that utilizes BERT (Bidirectional Encoder Representations from Transformers) to generate word embeddings and performs sequence classification.
- Random Forest: An ensemble learning method that combines multiple decision trees to make predictions.
- Decision Tree: A supervised machine learning algorithm that creates a tree-like model of decisions and their possible consequences.

Each model underwent specific preprocessing steps, such as text cleaning and vectorization techniques, to prepare the data for training and evaluation.

## Evaluation Metrics

The performance of each model was evaluated using the following metrics:

- Accuracy: The overall accuracy of the model in correctly predicting sentiment.
- Precision: The proportion of true positive predictions out of all positive predictions made by the model.
- Recall: The proportion of true positive predictions out of all actual positive instances in the dataset.
- F1-score: A measure that combines precision and recall, providing a balanced evaluation metric for binary classification.

The comparative analysis allows for an assessment of the strengths and weaknesses of each model in the context of sentiment analysis.

