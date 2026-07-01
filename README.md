# 🛍️ Retail Customer Feedback Intelligence Platform

<p align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

---

## 📌 Overview

The **Retail Customer Feedback Intelligence Platform** is an end-to-end Natural Language Processing (NLP) solution designed to transform unstructured customer reviews into actionable business insights.

Instead of simply classifying sentiment, the platform extracts retail entities, identifies product aspects, summarizes customer feedback, and generates executive-level business insights through an interactive analytics dashboard.

The project demonstrates how Machine Learning, Deep Learning, and NLP can be applied to solve real-world retail analytics problems at enterprise scale.

---

# 🚀 Features

### 🤖 NLP Models

- BERT Sentiment Analysis
- Retail Named Entity Recognition (NER)
- Aspect-Based Sentiment Analysis

### 📊 Analytics

- Business Insight Engine
- Executive Summary Generator
- Product-level Insights
- Category-level Insights
- Aspect Frequency Analysis

### 🌐 Applications

- FastAPI REST APIs
- Interactive Streamlit Dashboard
- Batch CSV Processing
- Single Review Analysis

---

# 🏗️ Solution Architecture

```text
                   Customer Reviews
                          │
                          ▼
                Data Preprocessing
                          │
                          ▼
            BERT Sentiment Classification
                          │
                          ▼
           Retail Entity Recognition (NER)
                          │
                          ▼
             Aspect Sentiment Extraction
                          │
                          ▼
              Business Insight Engine
                          │
                          ▼
              Executive Summarization
                          │
                          ▼
        FastAPI + Streamlit Dashboard
```

---

# 📂 Project Structure

```text
Retail_Customer_Feedback_Intelligence_Platform/

│
├── artifacts/
│
├── datasets/
│
├── src/
│   ├── analytics/
│   ├── api/
│   ├── models/
│   ├── nlp/
│   ├── pipeline/
│   ├── preprocessing/
│   └── training/
│
├── streamlit_app/
│   ├── app.py
│   ├── assets/
│   ├── components/
│   ├── config.py
│   └── views/
│
├── requirements.txt
│
└── README.md
```

---

# 🧠 Machine Learning Pipeline

### Sentiment Analysis

- Model

```
bert-base-uncased
```

- Framework

```
HuggingFace Transformers
```

- Backend

```
PyTorch
```

---

### Retail Named Entity Recognition

Extracts entities such as

- Product
- Department
- Category
- Subcategory

---

### Aspect Analysis

Automatically identifies customer opinions related to:

- Product Quality
- Battery
- Price
- Delivery
- Packaging
- Customer Service

---

# 📊 Business Insights

The platform automatically generates

- Overall Sentiment Distribution
- Rating Distribution
- Top Positive Products
- Top Negative Products
- Category Insights
- Aspect Frequency
- Executive Summary

---

# 🖥️ Streamlit Dashboard

The application includes

### 🏠 Home

Project overview and architecture.

### 📝 Review Analysis

Analyze individual customer reviews.

### 📊 Dashboard

Upload CSV files and generate business insights.

---

# 🌐 REST APIs

## Analyze Single Review

```
POST /analyze
```

Returns

- Sentiment
- Confidence
- Entities
- Aspect Sentiment

---

## Dashboard Analytics

```
POST /dashboard
```

Returns

- KPI Summary
- Charts
- Product Insights
- Category Insights
- Executive Summary

---

# ⚙️ Technology Stack

### Languages

- Python

### Machine Learning

- PyTorch
- HuggingFace Transformers
- Scikit-Learn

### NLP

- BERT
- Retail NER
- Aspect Extraction

### Backend

- FastAPI
- Pydantic

### Frontend

- Streamlit
- Plotly

### Data

- Pandas
- NumPy

---

# 📷 Screenshots

## Home

```
docs/images/home.png
```

## Review Analysis

```
docs/images/review_analysis.png
```

## Dashboard

```
docs/images/dashboard.png
```

## Swagger APIs

```
docs/images/swagger.png
```

---

# ▶️ Running the Project

## Clone

```bash
git clone https://github.com/<your-username>/Retail-Customer-Feedback-Intelligence-Platform.git
```

---

## Install

```bash
pip install -r requirements.txt
```

---

## Start FastAPI

```bash
uvicorn src.api.main:app --reload
```

Swagger

```
http://localhost:8000/docs
```

---

## Start Streamlit

```bash
streamlit run streamlit_app/app.py
```

---

# 📈 Future Enhancements

- LLM-based Review Summarization
- Model Performance Dashboard
- Docker Deployment
- Cloud Deployment
- Authentication
- Real-time Streaming Analytics

---

# 👨‍💻 Author

**Prakash**

Senior Data Scientist

Specializing in

- Machine Learning
- NLP
- Generative AI
- Retail Analytics

---

# ⭐ If you found this project useful

Please consider giving it a ⭐ on GitHub.