# 🧠 Mental Health Risk Signal Detector

A Streamlit-based **NLP risk signal detection system** that analyzes social media-style text and identifies potential mental health risk signals using machine learning.

🔗 **Live App:** https://your-app-name.streamlit.app  
🔗 **GitHub Repo:** https://github.com/YOUR_USERNAME/mental-health-risk-detector  

---

## 🚨 Problem

Individuals often express distress signals online before intervention.  
However, detecting these signals is challenging due to:

- subtle language patterns  
- overlapping emotional expressions  
- noisy and ambiguous labels  

This project builds a **triage-style system** that helps flag potential high-risk posts for **human review**, rather than attempting clinical diagnosis.

---

## 💡 Solution

This system uses:

- **TF-IDF + Logistic Regression**
- **Probability-based risk scoring**
- **Threshold optimization for safety-sensitive detection**
- **Human-in-the-loop review mechanism**

It provides:

- predicted class  
- risk score (0–1)  
- uncertainty estimate  
- review recommendation  

---

## ⚙️ System Design

### Two Operating Modes

#### 1. Balanced Mode (Threshold = 0.50)
- Optimized for overall performance  
- Accuracy ≈ 70%  
- Balanced precision and recall  

#### 2. Safety Mode (Threshold = 0.25)
- Optimized for **high recall (≈ 0.95)**  
- Minimizes missed high-risk cases  
- Accepts more false positives (reviewed by humans)

---

## 📊 Key Insight

Lowering the decision threshold significantly improves detection of high-risk signals:

> Recall increased to ~0.95 for high-risk posts, reducing missed cases at the cost of increased false positives.

This reflects **real-world risk detection systems**, where missing a critical case is more costly than raising an alert.

---

## 🧾 Features

### 🔎 Single Text Analysis
- Real-time prediction
- Risk score and classification
- Uncertainty and review flag

### 📂 Batch Upload
- Upload CSV with multiple posts
- Score all entries at once
- Download results

### 📊 Visualization
- Risk distribution (Low / Moderate / High)
- Confusion matrix comparison

### 🧠 Explainability
- Top predictive words influencing decisions

---

## 📁 Input Format (Batch Mode)

CSV file must include:

```csv
text
I feel tired of everything
I do not know how much longer I can keep going

