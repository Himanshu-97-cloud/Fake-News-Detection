# 📰 Fake News Detection using Hybrid ML Model

## 📌 Overview

This project implements a hybrid machine learning approach for fake news detection by combining:

* Transformer embeddings (DeBERTa)
* TF-IDF features
* Metadata features

The final classification is performed using a Random Forest model.

---

## 🚀 Features

* NLP using transformer models (DeBERTa)
* TF-IDF text vectorization
* Metadata-based feature engineering
* Hybrid ML pipeline

---

## ⚙️ Setup Instructions

### 1. Install dependencies

```bash
pip install pandas numpy torch transformers scikit-learn
```

### 2. Add dataset files

Place these files in project folder:

* train.tsv
* valid.tsv
* test.tsv

---

### 3. Run the project

```bash
python Deberta_TFIDF.py
```

---

## 📊 Output

* Accuracy
* Precision
* Recall
* F1 Score

---

## 👨‍💻 Author

Himanshu Pal
