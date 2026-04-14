🧠 Explainable AI for Mental Health Risk Detection (Text & Speech)<br>

An AI-powered application that analyzes text and voice inputs to identify potential signals of emotional distress and provide clear, interpretable risk insights.

⸻

🚀 Live Demo<br>

👉 https://mental-health-risk-detector-b5skfn3gxfdrimhz5f8hrr.streamlit.app/<br>

⸻

🎯 Overview<br>

This project demonstrates how machine learning and AI can be combined to move from raw predictions to interpretable, human-centered decision support.<br>

The system analyzes user input and classifies it into:<br>
    •    🟢 Neither / Other (no clear concern)<br>
    •    🟡 Depression-related signals<br>
    •    🔴 Suicide-related signals<br>

For each input, the system provides:<br>
    •    Risk score<br>
    •    Risk level<br>
    •    Uncertainty (confidence)<br>
    •    Human-review recommendation<br>
    •    AI-generated explanation<br>

⸻

🧠 Key Features<br>

🔎 Single Input Analysis<br>
    •    Analyze individual text inputs<br>
    •    Get real-time predictions and explanations<br>

🎤 Voice Input<br>
    •    Record speech directly in the browser<br>
    •    Automatic speech-to-text transcription<br>
    •    Instant analysis of spoken input<br>

📂 Batch Processing<br>
    •    Upload multiple inputs (CSV)<br>
    •    Analyze all records at once<br>
    •    Download results<br>
    •    View risk distribution<br>

🧠 AI Explanation Layer<br>
    •    Converts model outputs into plain English explanations<br>
    •    Highlights detected emotional signals<br>
    •    Suggests appropriate follow-up action<br>

⚙️ Guardrail System (Important)<br>
    •    Filters out non-mental-health text<br>
    •    Reduces false positives<br>
    •    Introduces a Neither / Other category<br>

⸻

🧪 Modeling Approach<br>
    •    Feature Engineering: TF-IDF (n-grams)<br>
    •    Model: Logistic Regression<br>
    •    Classes:<br>
    •    Depression-related<br>
    •    Suicide-related<br>
    •    Enhancement: Rule-based guardrail for “Neither / Other”<br>

Prediction Modes:<br>
    •    Balanced Mode → Standard classification<br>
    •    Safety Mode → Higher sensitivity for risk detection<br>

⸻

🧠 AI Integration<br>

This project combines:<br>
    •    Traditional Machine Learning (classification)<br>
    •    Explainable AI (interpretation layer)<br>
    •    Speech-to-Text (voice input)<br>
    •    Generative AI (LLM-based explanations and summaries)<br>

⸻

🧰 Tech Stack<br>
    •    Python<br>
    •    Scikit-learn<br>
    •    Pandas / NumPy<br>
    •    Streamlit<br>
    •    OpenAI API (speech + explanations)<br>

⸻

⚠️ Responsible Use<br>

This is a research and decision-support tool, not a clinical system.<br>

It is not intended for:<br>
    •    diagnosis<br>
    •    treatment decisions<br>
    •    automated interventions<br>

It is designed to:<br>
    •    support awareness<br>
    •    assist human review<br>
    •    demonstrate responsible AI design<br>

⸻

📉 Limitations<br>
    •    Binary training data (Depression vs SuicideWatch)<br>
    •    “Neither / Other” handled via rule-based guardrails<br>
    •    Text-only analysis (no personal context)<br>
    •    Possible false positives and false negatives<br>
    •    AI explanations are supportive, not ground truth<br>

⸻

🚀 Future Improvements<br>
    •    Train a true 3-class model (add “Neither / Other” data)<br>
    •    Incorporate transformer-based models (e.g., BERT)<br>
    •    Improve uncertainty calibration<br>
    •    Real-time streaming transcription<br>
    •    User-level temporal analysis<br>

⸻

👤 Author<br>

Prince Appiah<br>
Ph.D. Data Science<br>
Machine Learning | NLP | Explainable AI | Applied Analytics<br>
