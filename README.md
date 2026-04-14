🧠 Explainable AI for Mental Health Risk Detection (Text & Speech)

An AI-powered application that analyzes text and voice inputs to identify potential signals of emotional distress and provide clear, interpretable risk insights.

⸻

🚀 Live Demo

👉 https://mental-health-risk-detector-b5skfn3gxfdrimhz5f8hrr.streamlit.app/

⸻

🎯 Overview

This project demonstrates how machine learning and AI can be combined to move from raw predictions to interpretable, human-centered decision support.

The system analyzes user input and classifies it into:
    •    🟢 Neither / Other (no clear concern)
    •    🟡 Depression-related signals
    •    🔴 Suicide-related signals

For each input, the system provides:
    •    Risk score
    •    Risk level
    •    Uncertainty (confidence)
    •    Human-review recommendation
    •    AI-generated explanation

⸻

🧠 Key Features

🔎 Single Input Analysis
    •    Analyze individual text inputs
    •    Get real-time predictions and explanations

🎤 Voice Input
    •    Record speech directly in the browser
    •    Automatic speech-to-text transcription
    •    Instant analysis of spoken input

📂 Batch Processing
    •    Upload multiple inputs (CSV)
    •    Analyze all records at once
    •    Download results
    •    View risk distribution

🧠 AI Explanation Layer
    •    Converts model outputs into plain English explanations
    •    Highlights detected emotional signals
    •    Suggests appropriate follow-up action

⚙️ Guardrail System (Important)
    •    Filters out non-mental-health text
    •    Reduces false positives
    •    Introduces a Neither / Other category

⸻

🧪 Modeling Approach
    •    Feature Engineering: TF-IDF (n-grams)
    •    Model: Logistic Regression
    •    Classes:
    •    Depression-related
    •    Suicide-related
    •    Enhancement: Rule-based guardrail for “Neither / Other”

Prediction Modes:
    •    Balanced Mode → Standard classification
    •    Safety Mode → Higher sensitivity for risk detection

⸻

🧠 AI Integration

This project combines:
    •    Traditional Machine Learning (classification)
    •    Explainable AI (interpretation layer)
    •    Speech-to-Text (voice input)
    •    Generative AI (LLM-based explanations and summaries)

⸻

🧰 Tech Stack
    •    Python
    •    Scikit-learn
    •    Pandas / NumPy
    •    Streamlit
    •    OpenAI API (speech + explanations)

⸻

⚠️ Responsible Use

This is a research and decision-support tool, not a clinical system.

It is not intended for:
    •    diagnosis
    •    treatment decisions
    •    automated interventions

It is designed to:
    •    support awareness
    •    assist human review
    •    demonstrate responsible AI design

⸻

📉 Limitations
    •    Binary training data (Depression vs SuicideWatch)
    •    “Neither / Other” handled via rule-based guardrails
    •    Text-only analysis (no personal context)
    •    Possible false positives and false negatives
    •    AI explanations are supportive, not ground truth

⸻

🚀 Future Improvements
    •    Train a true 3-class model (add “Neither / Other” data)
    •    Incorporate transformer-based models (e.g., BERT)
    •    Improve uncertainty calibration
    •    Real-time streaming transcription
    •    User-level temporal analysis

⸻

👤 Author

Prince Appiah
Ph.D. Data Science
Machine Learning | NLP | Explainable AI | Applied Analytics
