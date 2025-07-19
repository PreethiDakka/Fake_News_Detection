# 📰 Fake News Detection Using Machine Learning and Deep Learning

A machine learning and deep learning-based Fake News Detection system using SVM, Naive Bayes, and LSTM to classify news articles as real or fake.

This project aims to detect fake news using various machine learning and deep learning models such as **SVM**, **Naive Bayes**, and **LSTM**. The system classifies news articles as real or fake based on the content.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Technologies Used](#-technologies-used)
- [Dataset](#-dataset)
- [Model Overview](#-model-overview)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [To-Do](#-to-do)
- [License](#-license)

---

## 📖 Overview

With the rise of misinformation on the internet, detecting fake news is critical. This project tackles this issue by:
- Preprocessing news articles,
- Vectorizing text using **TF-IDF**, and
- Applying classification algorithms for real/fake prediction.

---

## 🧰 Technologies Used

- Python 3.x
- Scikit-learn
- Keras / TensorFlow
- NLTK
- Pandas, NumPy, Matplotlib
- Jupyter Notebook / Pickle (if deployed)

---

## 📊 Dataset

We use the [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data), which contains news articles with corresponding labels (1 = fake, 0 = real).

> You can download the dataset manually and place it inside the `dataset/` folder.

---

## 🧠 Model Overview

| Model         | Description                                      |
|---------------|--------------------------------------------------|
| SVM           | High-accuracy linear classifier using TF-IDF     |
| Naive Bayes   | Fast and efficient baseline classifier           |
| LSTM          | Deep learning model trained on tokenized text    |

---
