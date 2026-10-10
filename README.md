# SMS Spam Detection Using Machine Learning

## Project Overview

This project uses Machine Learning to classify SMS messages as Spam or Ham (legitimate messages).

It uses CountVectorizer for text vectorization and Naive Bayes algorithms for classification.

## Objectives

- Classify SMS messages as Spam or Ham.
- Perform Exploratory Data Analysis (EDA).
- Remove duplicate records.
- Convert text messages into numerical features.
- Train and compare Naive Bayes classifiers.
- Evaluate model performance.
- Analyze incorrectly classified messages.
- Save and load the trained model.
- Predict categories for new SMS messages.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- VS Code

## Dataset

The project uses the SMSSpamCollection dataset.

### Dataset Columns

- `label`: Indicates whether the message is Ham or Spam.
- `message`: Contains the original SMS text.

### Target Encoding

- `ham` = 0
- `spam` = 1

## Project Workflow

### 1. Data Loading

The dataset is loaded using Pandas. The file is tab-separated and contains SMS labels and messages.

### 2. Exploratory Data Analysis (EDA)

The dataset is analyzed to check:

- Dataset dimensions
- Data types
- Column names
- Missing values
- Duplicate records
- Descriptive statistics

### 3. Data Cleaning

Duplicate rows are removed using Pandas.

### 4. Feature and Target Separation

- `X`: Contains SMS messages.
- `y`: Contains numerical target labels.

### 5. Train-Test Split

The dataset is divided into training and testing sets.

- Training data: 80%
- Testing data: 20%
- Random state: 42

### 6. Text Vectorization

CountVectorizer converts SMS messages into numerical features.

The vectorizer learns its vocabulary from the training messages and transforms the training and testing messages using the same vocabulary.

### 7. Model Training and Comparison

Three Naive Bayes classifiers were evaluated:

- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Bernoulli Naive Bayes

Bernoulli Naive Bayes was selected as the final model based on the previously recorded comparison results.

### 8. Final Model Configuration

- Model: Bernoulli Naive Bayes
- Alpha: 0.01
- Binarize: 0

## Model Evaluation

The model is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### Previously Recorded Model Comparison

| Model | Previously Recorded Test F1-Score |
|---|---:|
| Gaussian Naive Bayes | 72.73% |
| Multinomial Naive Bayes | 91.29% |
| Bernoulli Naive Bayes | 95.65% |

### Latest Recorded BernoulliNB Results

| Metric | Training Set | Testing Set |
|---|---:|---:|
| F1-Score | 99.41% | 94.81% |
| Precision | 100.00% | 98.46% |
| Accuracy | 99.85% | 98.65% |
| Recall | 98.83% | 91.43% |

Note: The model comparison scores are from earlier experiments. The latest results reflect the evaluation after correcting the vectorization process.

## Error Analysis

Error analysis identifies messages that the model classified incorrectly.

The project compares actual labels with predicted labels and displays the original SMS text for misclassified messages.

### Types of Errors

- False Positive: A Ham message is incorrectly classified as Spam.
- False Negative: A Spam message is incorrectly classified as Ham.

## Model Saving and Loading

The trained model and fitted vectorizer are saved using Joblib.

### Saved Files

- `Model_Bernoullis.pkl`
- `Vectorise.pkl`

Both files are loaded when making predictions on new SMS messages.

## Making Predictions on New Messages

The saved CountVectorizer transforms new SMS text into numerical features. The saved Bernoulli Naive Bayes model predicts whether the message is Spam or Ham.

### Example Input

`Congratulations! You won a free prize`

The program prints the predicted category: `Spam` or `Ham`.

## Project Structure

```text
SMS-Spam-Detection/
├── SMSSpamCollection
├── SMS_Spam_Detection.py
├── Model_Bernoullis.pkl
├── Vectorise.pkl
└── README.md