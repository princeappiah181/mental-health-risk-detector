🧠 Mental Health Risk Signal Detector<br>

A Streamlit-based NLP risk signal detection system that analyzes social media-style text and identifies potential mental health risk signals using machine learning.<br>

🔗 Live App: https://your-app-name.streamlit.app<br>
🔗 GitHub Repo: https://github.com/YOUR_USERNAME/mental-health-risk-detector<br>

⸻

🚨 Problem<br>

Individuals often express distress signals online before intervention.<br>
However, detecting these signals is challenging due to:<br>
    •    subtle language patterns<br>
    •    overlapping emotional expressions<br>
    •    noisy and ambiguous labels<br>

This project builds a triage-style system that helps flag potential high-risk posts for human review, rather than attempting clinical diagnosis.<br>

⸻

💡 Solution<br>

This system uses:<br>
    •    TF-IDF + Logistic Regression<br>
    •    Probability-based risk scoring<br>
    •    Threshold optimization for safety-sensitive detection<br>
    •    Human-in-the-loop review mechanism<br>

It provides:<br>
    •    predicted class<br>
    •    risk score (0–1)<br>
    •    uncertainty estimate<br>
    •    review recommendation<br>

⸻

⚙️ System Design<br>

Two Operating Modes<br>

1. Balanced Mode (Threshold = 0.50)<br>
    •    Optimized for overall performance<br>
    •    Accuracy ≈ 70%<br>
    •    Balanced precision and recall<br>

2. Safety Mode (Threshold = 0.25)<br>
    •    Optimized for high recall (≈ 0.95)<br>
    •    Minimizes missed high-risk cases<br>
    •    Accepts more false positives (reviewed by humans)<br>

⸻

📊 Key Insight<br>

Lowering the decision threshold significantly improves detection of high-risk signals:<br>

Recall increased to ~0.95 for high-risk posts, reducing missed cases at the cost of increased false positives.<br>

This reflects real-world risk detection systems, where missing a critical case is more costly than raising an alert.<br>

⸻

🧾 Features<br>

🔎 Single Text Analysis<br>
    •    Real-time prediction<br>
    •    Risk score and classification<br>
    •    Uncertainty and review flag<br>

📂 Batch Upload<br>
    •    Upload CSV with multiple posts<br>
    •    Score all entries at once<br>
    •    Download results<br>

📊 Visualization<br>
    •    Risk distribution (Low / Moderate / High)<br>
    •    Confusion matrix comparison<br>

🧠 Explainability<br>
    •    Top predictive words influencing decisions<br>

⸻

📁 Input Format (Batch Mode)<br>

text<br>
I feel tired of everything<br>
I do not know how much longer I can keep going<br>

🧪 Tech Stack<br>
    •    Python<br>
    •    scikit-learn (TF-IDF + Logistic Regression)<br>
    •    pandas / numpy (data processing)<br>
    •    Streamlit (interactive web app)<br>

⸻

⚠️ Ethical Considerations<br>

This project is a research prototype for risk signal detection, not a clinical system.<br>

Not intended for:<br>
    •    diagnosis<br>
    •    treatment decisions<br>
    •    automated intervention<br>

Safeguards implemented:<br>
    •    uncertainty-based review mechanism<br>
    •    human-in-the-loop design<br>
    •    transparent model behavior and outputs<br>

⸻

📉 Limitations<br>
    •    Label ambiguity (depression vs SuicideWatch overlap)<br>
    •    Text-only analysis (no user context or history)<br>
    •    False positives and false negatives remain<br>
    •    Model performance depends on dataset assumptions<br>

⸻

🚀 Future Improvements<br>
    •    Transformer-based models (e.g., BERT)<br>
    •    Temporal modeling of user activity<br>
    •    More robust uncertainty estimation<br>
    •    Pre-trained model deployment for faster inference<br>

▶️ Run Locally<br>

pip install -r requirements.txt<br>
streamlit run app.py<br>

👤 Author<br>
Prince Appiah<br>
PhD Data Science<br>
