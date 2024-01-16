# Fake News Classifier

## Overview

This project focuses on building a machine learning model to classify news articles as either fake or real. The classifier is trained on a dataset containing labeled examples of fake and real news articles, and it leverages the Multinomial Naive Bayes algorithm.

## Project Structure

- **Notebook:** The code is implemented in a Jupyter notebook using Python. The notebook covers data loading, preprocessing, feature extraction using CountVectorizer, model training using Multinomial Naive Bayes, and evaluation.

- **Dataset:** The dataset used is named "train.csv" and is stored locally at "D:/NLP/Fake_news_classifier/". Ensure the dataset is available at this location for the code to execute properly.

## Getting Started

To run the notebook and reproduce the results, follow these steps:

1. Install required libraries:

    ```python
    !pip install pandas scikit-learn matplotlib nltk
    ```

2. Execute the notebook cells sequentially.

## Data Preprocessing

- Loaded the dataset from "train.csv" and extracted independent (X) and dependent (y) features.
- Applied text preprocessing techniques, including removing stopwords, stemming, and converting text to lowercase.

## Feature Extraction

- Utilized CountVectorizer to convert text data into a bag-of-words model with a maximum of 5000 features and n-gram range (1,3).

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()
```

## Model Building

- Implemented a Multinomial Naive Bayes classifier from the scikit-learn library.
- Trained the classifier on the training data and evaluated its performance on the test set.

```python
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
```

## Model Evaluation

- Assessed the model's accuracy and visualized the confusion matrix using a custom function.

```python
score = metrics.accuracy_score(y_test, pred)
print("Accuracy:  %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])
```

## Conclusion

This project demonstrates the use of machine learning, specifically the Multinomial Naive Bayes algorithm, for classifying news articles as fake or real. The code provided in the notebook can be used as a starting point for building more sophisticated models or experimenting with different algorithms.

