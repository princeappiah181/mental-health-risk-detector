#!/usr/bin/env python
# coding: utf-8

import os
import re
import string
import math
import json
import tempfile
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from pydub import AudioSegment
from st_audiorec import st_audiorec

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Mental Health Risk Signal Detector",
    page_icon="🧠",
    layout="wide"
)

# =========================================================
# SESSION STATE
# =========================================================
if "prediction_log" not in st.session_state:
    st.session_state.prediction_log = pd.DataFrame(
        columns=[
            "mode",
            "original_text",
            "cleaned_text",
            "predicted_class",
            "predicted_label",
            "risk_score",
            "uncertainty",
            "needs_review",
            "risk_level",
            "guardrail_reason",
        ]
    )

if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""

if "example_text" not in st.session_state:
    st.session_state.example_text = ""

# =========================================================
# OPENAI CLIENT
# =========================================================
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY. Add it in Streamlit secrets.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================================================
# TEXT CLEANING
# =========================================================
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

# =========================================================
# NEITHER / OTHER GUARDRAIL
# =========================================================
def detect_neither_other(text: str):
    text_l = clean_text(text)

    exact_non_mh_phrases = {
        "fuck around and find out",
        "fafo",
        "lol",
        "lmao",
        "good game",
        "what a game",
        "lets go",
        "let's go",
        "he is cooked",
        "she is cooked",
        "wtf",
        "bruh",
    }

    mh_cues = [
        "depressed", "depression", "suicidal", "suicide",
        "want to die", "kill myself", "end my life",
        "hurt myself", "self harm", "selfharm",
        "hopeless", "empty", "worthless", "alone",
        "cant go on", "cannot go on", "give up on life",
        "tired of life", "dont want to live", "do not want to live",
        "nobody cares", "no reason to live", "i want to disappear",
        "i hate my life", "i hate myself", "end it all",
    ]

    softer_distress_cues = [
        "anxious", "anxiety", "panic", "overwhelmed",
        "exhausted", "burned out", "burnt out",
        "sad", "crying", "lonely", "miserable",
        "stressed", "stress", "ashamed", "guilty",
    ]

    aggressive_markers = [
        "fuck", "fucked", "wtf", "find out", "idiot",
        "stupid", "shut up", "fight me", "come at me",
    ]

    positive_or_neutral_markers = [
        "good", "great", "happy", "fun", "food", "drinks",
        "enjoyed", "nice", "beautiful", "love this", "awesome",
        "amazing", "had a good day", "today was good"
    ]

    if text_l in exact_non_mh_phrases:
        return {
            "is_neither": True,
            "label": "Neither / Other",
            "reason": "Common slang or non-mental-health phrase detected without clear depression or self-harm cues."
        }

    has_strong_mh = any(cue in text_l for cue in mh_cues)
    has_soft_distress = any(cue in text_l for cue in softer_distress_cues)
    has_aggressive = any(marker in text_l for marker in aggressive_markers)
    has_positive_neutral = any(marker in text_l for marker in positive_or_neutral_markers)

    # Strongest rule:
    # if no mental-health cues at all, classify as Neither / Other
    if not has_strong_mh and not has_soft_distress:
        return {
            "is_neither": True,
            "label": "Neither / Other",
            "reason": "No clear depression, suicide, or emotional-distress cues detected."
        }

    # Very short text with no MH cues
    if len(text_l.split()) <= 6 and not has_strong_mh and not has_soft_distress:
        return {
            "is_neither": True,
            "label": "Neither / Other",
            "reason": "Short text with no clear mental-health-related cues."
        }

    # Aggressive but not MH-related
    if has_aggressive and not has_strong_mh and not has_soft_distress:
        return {
            "is_neither": True,
            "label": "Neither / Other",
            "reason": "Aggressive or confrontational language detected without clear distress or self-harm signals."
        }

    # Positive/neutral everyday text without MH cues
    if has_positive_neutral and not has_strong_mh and not has_soft_distress:
        return {
            "is_neither": True,
            "label": "Neither / Other",
            "reason": "Neutral or positive everyday language detected without clear mental-health-related signals."
        }

    return {
        "is_neither": False,
        "label": None,
        "reason": None
    }
# =========================================================
# HELPERS
# =========================================================
def uncertainty_score(score: float) -> float:
    return 1 - abs(score - 0.5) * 2

def risk_level_balanced(score: float) -> str:
    if score < 0.30:
        return "Low"
    elif score < 0.70:
        return "Moderate"
    return "High"

def needs_review_balanced(score: float) -> bool:
    return 0.40 <= score <= 0.60

def risk_level_safety(score: float) -> str:
    if score < 0.25:
        return "Low"
    elif score < 0.60:
        return "Moderate / Review"
    return "High"

def needs_review_safety(score: float) -> bool:
    return 0.25 <= score <= 0.60

def predict_scores(text_list, vectorizer, model, threshold, mode_name):
    cleaned_texts = [clean_text(t) for t in text_list]
    vec = vectorizer.transform(cleaned_texts)
    probs = model.predict_proba(vec)[:, 1]
    preds = (probs >= threshold).astype(int)

    records = []
    for original_text, cleaned_text, prob, pred in zip(text_list, cleaned_texts, probs, preds):
        unc = uncertainty_score(prob)

        if mode_name == "Balanced Mode":
            level = risk_level_balanced(prob)
            review = needs_review_balanced(prob)
        else:
            level = risk_level_safety(prob)
            review = needs_review_safety(prob)

        label_name = "SuicideWatch-related" if pred == 1 else "Depression-related"

        records.append({
            "mode": mode_name,
            "original_text": original_text,
            "cleaned_text": cleaned_text,
            "predicted_class": label_name,
            "predicted_label": int(pred),
            "risk_score": float(prob),
            "uncertainty": float(unc),
            "needs_review": bool(review),
            "risk_level": level,
            "guardrail_reason": "",
        })

    return pd.DataFrame(records)

def predict_with_guardrail(text_list, vectorizer, model, threshold, mode_name):
    """
    First applies the Neither / Other guardrail.
    If the text is relevant, falls back to the binary classifier.
    """
    records = []

    for text in text_list:
        guardrail = detect_neither_other(text)

        if guardrail["is_neither"]:
            cleaned = clean_text(text)
            records.append({
                "mode": mode_name,
                "original_text": text,
                "cleaned_text": cleaned,
                "predicted_class": "Neither / Other",
                "predicted_label": -1,
                "risk_score": 0.05,
                "uncertainty": 0.10,
                "needs_review": False,
                "risk_level": "Low",
                "guardrail_reason": guardrail["reason"],
            })
        else:
            pred_df = predict_scores([text], vectorizer, model, threshold, mode_name).copy()
            records.append(pred_df.iloc[0].to_dict())

    return pd.DataFrame(records)

# =========================================================
# AI EXPLANATION LAYER
# =========================================================
def build_action_label(risk_level: str, needs_review: bool, predicted_class: str = "") -> str:
    if predicted_class == "Neither / Other":
        return "No mental-health follow-up indicated"
    if risk_level == "High":
        return "Urgent review recommended"
    if needs_review:
        return "Review recommended"
    if risk_level in ["Moderate", "Moderate / Review"]:
        return "Monitor"
    return "Low concern"

def generate_ai_explanation(
    raw_text: str,
    predicted_class: str,
    risk_score: float,
    risk_level: str,
    needs_review: bool,
    mode_name: str
):
    suggested_action = build_action_label(risk_level, needs_review, predicted_class)

    prompt = f"""
You are assisting with a research prototype for mental health risk signal triage.
This is NOT a diagnosis tool. Do not make clinical claims.

Given the post below and the model outputs, provide a brief structured interpretation.

Post:
\"\"\"{raw_text}\"\"\"

Model outputs:
- Predicted class: {predicted_class}
- Risk score: {risk_score:.3f}
- Risk level: {risk_level}
- Needs review: {needs_review}
- Operating mode: {mode_name}
- Suggested action baseline: {suggested_action}

Return ONLY valid JSON with this schema:
{{
  "explanation": "2-4 sentences, plain English, concise",
  "detected_signals": ["signal1", "signal2", "signal3"],
  "suggested_action": "one short action label"
}}

Rules:
- Do not claim diagnosis.
- If the text looks like slang, aggression, humor, or non-mental-health language, say that clearly.
- If the predicted class is "Neither / Other", explain that the text does not show clear depression or suicide-related cues.
- Focus on language cues such as hopelessness, distress, isolation, fatigue, self-harm references, uncertainty, emotional overwhelm.
- If the text does not strongly indicate severe concern, say so.
- Keep detected_signals short phrases.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a careful assistant that returns only valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )

        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)

        explanation = parsed.get("explanation", "No explanation generated.")
        detected_signals = parsed.get("detected_signals", [])
        suggested_action = parsed.get("suggested_action", suggested_action)

        if not isinstance(detected_signals, list):
            detected_signals = [str(detected_signals)]

        return {
            "explanation": explanation,
            "detected_signals": detected_signals,
            "suggested_action": suggested_action,
        }

    except Exception:
        fallback_signals = []
        lower_text = raw_text.lower()
        for word in ["hopeless", "empty", "tired", "alone", "give up", "cant go on", "hurt", "die", "suicide"]:
            if word in lower_text:
                fallback_signals.append(word)

        if predicted_class == "Neither / Other":
            fallback_signals = ["non-mental-health language"]
            fallback_explanation = (
                "The input appears to be slang, confrontational language, or otherwise outside clear depression or suicide-related cues. "
                "It was therefore labeled as Neither / Other."
            )
        else:
            if not fallback_signals:
                fallback_signals = ["emotional distress cues"]
            fallback_explanation = (
                "The post was interpreted using the model's predicted class, risk score, and review status. "
                "This result should be treated as a research-style triage signal rather than a diagnosis."
            )

        return {
            "explanation": fallback_explanation,
            "detected_signals": fallback_signals[:3],
            "suggested_action": suggested_action,
        }

def generate_batch_ai_summary(final_batch: pd.DataFrame):
    try:
        high_count = int((final_batch["risk_level"] == "High").sum())
        review_count = int(final_batch["needs_review"].sum())
        neither_count = int((final_batch["predicted_class"] == "Neither / Other").sum())
        total_count = int(len(final_batch))

        sample_rows = final_batch.head(20).copy()

        example_lines = []
        for _, row in sample_rows.iterrows():
            raw_text = str(row.get("text", row.get("original_text", "")))[:300]
            example_lines.append(
                f"- Text: {raw_text}\n"
                f"  Predicted class: {row.get('predicted_class', '')}\n"
                f"  Risk level: {row.get('risk_level', '')}\n"
                f"  Risk score: {row.get('risk_score', '')}\n"
                f"  Needs review: {row.get('needs_review', '')}"
            )

        joined_examples = "\n".join(example_lines)

        prompt = f"""
You are assisting with a research prototype for mental health risk triage.
This is NOT a diagnosis system. Do not make clinical claims.

Below is a batch of analyzed posts.

Batch stats:
- Total posts: {total_count}
- Neither / Other count: {neither_count}
- High risk count: {high_count}
- Needs review count: {review_count}

Examples:
{joined_examples}

Return ONLY valid JSON with this schema:
{{
  "summary": "3-5 sentence summary of the overall batch",
  "dominant_signals": ["signal1", "signal2", "signal3"],
  "follow_up_focus": "one short paragraph with recommended review focus"
}}

Rules:
- Keep the language non-clinical
- Mention if many inputs appear unrelated to mental-health signals
- Focus on patterns such as hopelessness, emotional distress, fatigue, uncertainty, isolation, self-harm language
- Do not overstate certainty
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a careful assistant that returns only valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )

        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)

        return {
            "summary": parsed.get("summary", "No summary generated."),
            "dominant_signals": parsed.get("dominant_signals", []),
            "follow_up_focus": parsed.get("follow_up_focus", "No follow-up focus generated."),
        }

    except Exception:
        high_count = int((final_batch["risk_level"] == "High").sum())
        review_count = int(final_batch["needs_review"].sum())
        neither_count = int((final_batch["predicted_class"] == "Neither / Other").sum())
        total_count = int(len(final_batch))

        return {
            "summary": (
                f"This batch contains {total_count} posts, with {neither_count} labeled as Neither / Other, "
                f"{high_count} classified as high risk, and {review_count} flagged for review."
            ),
            "dominant_signals": ["emotional distress", "hopelessness", "review-needed language"],
            "follow_up_focus": (
                "Prioritize posts classified as High and then examine posts flagged for review. "
                "Separate clearly unrelated/slang content from genuine distress-related language."
            ),
        }

# =========================================================
# AUDIO / TRANSCRIPTION HELPERS
# =========================================================
def wav_bytes_to_audiosegment(wav_bytes: bytes) -> AudioSegment:
    return AudioSegment.from_file(BytesIO(wav_bytes), format="wav")

def split_audio_into_chunks(audio_segment: AudioSegment, chunk_ms: int = 60_000):
    chunks = []
    total_ms = len(audio_segment)
    num_chunks = math.ceil(total_ms / chunk_ms)

    for i in range(num_chunks):
        start = i * chunk_ms
        end = min((i + 1) * chunk_ms, total_ms)
        chunks.append(audio_segment[start:end])

    return chunks

def transcribe_audiosegment_with_openai(
    audio_segment: AudioSegment,
    model_name: str = "gpt-4o-mini-transcribe",
) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        audio_segment.export(tmp.name, format="wav")
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=model_name,
                file=audio_file,
            )
        return transcript.text if hasattr(transcript, "text") else str(transcript)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def transcribe_short_wav_bytes(
    wav_bytes: bytes,
    model_name: str = "gpt-4o-mini-transcribe",
) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(wav_bytes)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=model_name,
                file=audio_file,
            )
        return transcript.text if hasattr(transcript, "text") else str(transcript)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def transcribe_long_wav_bytes(
    wav_bytes: bytes,
    model_name: str = "gpt-4o-mini-transcribe",
    chunk_ms: int = 60_000,
) -> str:
    audio = wav_bytes_to_audiosegment(wav_bytes)
    chunks = split_audio_into_chunks(audio, chunk_ms=chunk_ms)

    if len(chunks) == 1:
        return transcribe_short_wav_bytes(wav_bytes, model_name=model_name)

    transcripts = []
    for chunk in chunks:
        text = transcribe_audiosegment_with_openai(chunk, model_name=model_name)
        transcripts.append(text.strip())

    return " ".join(t for t in transcripts if t)

# =========================================================
# LOAD SAVED ARTIFACTS
# =========================================================
@st.cache_resource
def load_artifacts():
    artifacts = joblib.load("artifacts/mental_health_model.joblib")
    return (
        artifacts["vectorizer"],
        artifacts["model"],
        artifacts["metrics"],
        artifacts["top_positive"],
        artifacts["top_negative"],
    )

vectorizer, model, metrics, top_positive, top_negative = load_artifacts()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("⚙️ Prediction Mode")
mode = st.sidebar.radio(
    "Choose system mode",
    ["Balanced Mode", "Safety Mode"],
)

if mode == "Balanced Mode":
    active_threshold = metrics["balanced"]["threshold"]
    st.sidebar.info(
        "Balanced Mode uses threshold = 0.50.\n\nBest for standard evaluation and balanced performance."
    )
else:
    active_threshold = metrics["safety"]["threshold"]
    st.sidebar.warning(
        "Safety Mode uses threshold = 0.25.\n\nBest for high recall and triage-style review."
    )

st.sidebar.markdown("---")
st.sidebar.subheader("About the labels")
st.sidebar.write("**-1** = Neither / Other")
st.sidebar.write("**0** = Depression-related")
st.sidebar.write("**1** = SuicideWatch-related")

# =========================================================
# MAIN HEADER
# =========================================================
st.title("🧠 Mental Health Risk Signal Detector")
st.caption(
    "An AI-powered system that analyzes text and voice to identify early signals of emotional distress and support risk-aware decision making."
)

st.markdown(
    """
This tool classifies input into:
- ** No concern (Neither / Other)**
- **Depression-related signals**
- **Suicide-related signals**

Each analysis includes:
- A **risk score**
- A **risk level**
- An **uncertainty-aware score**
- A **review recommendation**
- An **AI-generated explanation**


**Key Capabilities**
    - **Multi-modal input: supports both text and voice.**
    -  **Real-time analysis: instant classification and feedback.**
    -  **Explainable AI: translates model output into human-understandable insights.**
    -  **False-positive control: includes a Neither / Other guardrail to filter non-relevant language.**
    -  **Batch intelligence: summarizes patterns across multiple messages for triage-style workflows.**


**Why this matters**

People often express distress in subtle ways. This system demonstrates how AI can:
    -   Detect early warning signals in language
    -    Reduce missed high-risk cases
    -   Provide interpretable outputs for human review
    -    Bridge the gap between raw predictions and actionable insights


**⚠️ Responsible Use**

This is a research and decision-support tool, not a clinical system.
It is designed to assist awareness and human judgment, not replace professional evaluation.
"""
)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔎 Analyze Text",
    "🎤 Voice Input",
    "📂 Batch Upload",
    "📊 Model Performance",
    "🧾 Explainability",
    "⚠️ Ethics & Safeguards",
])

# =========================================================
# RESULT RENDERER
# =========================================================
def render_result_block(raw_text, result, mode_name):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Class", result["predicted_class"])
    c2.metric("Risk Score", f"{result['risk_score']:.3f}")
    c3.metric("Uncertainty", f"{result['uncertainty']:.3f}")
    c4.metric("Risk Level", result["risk_level"])

    if result["predicted_class"] == "Neither / Other":
        st.info("This input was identified as outside clear depression or suicide-related categories.")
    elif result["needs_review"]:
        st.warning("This input falls into the human-review zone for the selected mode.")
    else:
        st.success("This input is outside the human-review zone for the selected mode.")

    if result.get("predicted_class") == "Neither / Other" and result.get("guardrail_reason"):
        st.info(f"Guardrail decision: {result['guardrail_reason']}")

    ai_info = generate_ai_explanation(
        raw_text=raw_text,
        predicted_class=result["predicted_class"],
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        needs_review=result["needs_review"],
        mode_name=mode_name,
    )

    st.markdown("### 🧠 AI Explanation")
    st.write(ai_info["explanation"])

    st.markdown("### 🔎 Detected Signals")
    if ai_info["detected_signals"]:
        for sig in ai_info["detected_signals"]:
            st.write(f"- {sig}")
    else:
        st.write("- No clear signals extracted")

    st.markdown("### ⚠️ Suggested Action")
    st.info(ai_info["suggested_action"])

    with st.expander("View cleaned text used by the model"):
        st.write(result["cleaned_text"])

    st.markdown("### Interpretation")
    st.write(
        f"""
- **Mode:** {result['mode']}
- **Threshold used:** {active_threshold}
- **Predicted class:** {result['predicted_class']}
- **Risk score:** {result['risk_score']:.3f}
- **Risk level:** {result['risk_level']}
- **Needs review:** {result['needs_review']}
"""
    )

# =========================================================
# TAB 1: SINGLE TEXT ANALYSIS
# =========================================================
with tab1:
    st.subheader("Analyze a Single Post")

    sample_text = st.text_area(
        "Paste a post below",
        height=220,
        placeholder="Enter text here...",
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        run_btn = st.button("Run Analysis")

    with col2:
        if st.button("Load Example"):
            st.session_state.example_text = (
                "I feel tired of everything and I do not know how much longer I can keep going."
            )

    if st.session_state.example_text and not sample_text:
        sample_text = st.session_state.example_text

    if run_btn:
        if not sample_text.strip():
            st.error("Please enter some text.")
        else:
            result_df = predict_with_guardrail(
                [sample_text],
                vectorizer,
                model,
                active_threshold,
                mode,
            )
            result = result_df.iloc[0]

            render_result_block(sample_text, result, mode)

            st.session_state.prediction_log = pd.concat(
                [st.session_state.prediction_log, result_df],
                ignore_index=True,
            )

    st.markdown("### Download Prediction Log")
    if not st.session_state.prediction_log.empty:
        log_csv = st.session_state.prediction_log.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download prediction log as CSV",
            data=log_csv,
            file_name="prediction_log.csv",
            mime="text/csv",
        )
        st.dataframe(st.session_state.prediction_log.tail(10), use_container_width=True)

        if st.button("Clear Prediction Log"):
            st.session_state.prediction_log = st.session_state.prediction_log.iloc[0:0]
            st.success("Prediction log cleared.")
            st.rerun()
    else:
        st.info("No predictions logged yet.")

# =========================================================
# TAB 2: VOICE INPUT
# =========================================================
with tab2:
    st.subheader("Voice Input")

    st.markdown(
        """
Record speech directly in the browser and analyze it in one step.

This uses **gpt-4o-mini-transcribe** for faster transcription, then sends the transcript
to the classifier automatically.
"""
    )

    st.info("For best speed, keep recordings short, ideally 2–5 seconds.")

    wav_audio_data = st_audiorec()

    if wav_audio_data is not None:
        st.audio(wav_audio_data, format="audio/wav")

        if st.button("Analyze Voice"):
            try:
                with st.spinner("Transcribing and analyzing audio..."):
                    transcript = transcribe_short_wav_bytes(
                        wav_audio_data,
                        model_name="gpt-4o-mini-transcribe",
                    )

                    st.session_state.voice_transcript = transcript

                    result_df = predict_with_guardrail(
                        [transcript],
                        vectorizer,
                        model,
                        active_threshold,
                        mode,
                    )
                    result = result_df.iloc[0]

                render_result_block(transcript, result, mode)

                with st.expander("View transcript"):
                    st.write(transcript)

                st.session_state.prediction_log = pd.concat(
                    [st.session_state.prediction_log, result_df],
                    ignore_index=True,
                )

            except Exception as e:
                st.error(f"Voice analysis failed: {e}")

# =========================================================
# TAB 3: BATCH UPLOAD
# =========================================================
with tab3:
    st.subheader("Batch Score Multiple Posts")

    st.markdown(
        """
Upload a CSV file containing a text column.  
Recommended column name: **text**
"""
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        st.markdown("### Preview of Uploaded File")
        st.dataframe(batch_df.head(), use_container_width=True)

        if "text" not in batch_df.columns:
            st.error("Your CSV must contain a column named 'text'.")
        else:
            if st.button("Run Batch Scoring"):
                batch_df["text"] = batch_df["text"].fillna("").astype(str)

                batch_results = predict_with_guardrail(
                    batch_df["text"].tolist(),
                    vectorizer,
                    model,
                    active_threshold,
                    mode,
                )

                final_batch = pd.concat(
                    [batch_df.reset_index(drop=True), batch_results.drop(columns=["original_text"])],
                    axis=1,
                )

                st.markdown("### Batch Results")
                st.dataframe(final_batch.head(20), use_container_width=True)

                batch_csv = final_batch.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download batch results as CSV",
                    data=batch_csv,
                    file_name="batch_predictions.csv",
                    mime="text/csv",
                )

                st.markdown("### Batch Summary")
                summary_counts = final_batch["predicted_class"].value_counts().reset_index()
                summary_counts.columns = ["predicted_class", "count"]
                st.dataframe(summary_counts, use_container_width=True)

                st.markdown("### Risk Level Distribution")
                risk_summary = final_batch["risk_level"].value_counts().reset_index()
                risk_summary.columns = ["risk_level", "count"]

                risk_order = ["Low", "Moderate", "Moderate / Review", "High"]
                risk_summary["risk_level"] = pd.Categorical(
                    risk_summary["risk_level"],
                    categories=risk_order,
                    ordered=True,
                )
                risk_summary = risk_summary.sort_values("risk_level")

                chart_df = risk_summary.set_index("risk_level")
                st.bar_chart(chart_df["count"])

                st.markdown("### 🧠 AI Batch Summary")
                with st.spinner("Generating AI summary for batch results..."):
                    batch_ai_summary = generate_batch_ai_summary(final_batch)

                st.markdown("#### Summary")
                st.write(batch_ai_summary["summary"])

                st.markdown("#### Dominant Signals")
                if batch_ai_summary["dominant_signals"]:
                    for sig in batch_ai_summary["dominant_signals"]:
                        st.write(f"- {sig}")
                else:
                    st.write("- No dominant signals extracted")

                st.markdown("#### Suggested Follow-Up Focus")
                st.info(batch_ai_summary["follow_up_focus"])

# =========================================================
# TAB 4: MODEL PERFORMANCE
# =========================================================
with tab4:
    st.subheader("Performance Summary")

    summary_df = pd.DataFrame({
        "Mode": ["Balanced Mode", "Safety Mode"],
        "Threshold": [
            metrics["balanced"]["threshold"],
            metrics["safety"]["threshold"],
        ],
        "Accuracy": [
            metrics["balanced"]["accuracy"],
            metrics["safety"]["accuracy"],
        ],
        "F1 (Class 1)": [
            metrics["balanced"]["f1_class1"],
            metrics["safety"]["f1_class1"],
        ],
        "Recall (Class 1)": [
            metrics["balanced"]["report"]["1"]["recall"],
            metrics["safety"]["report"]["1"]["recall"],
        ],
        "Precision (Class 1)": [
            metrics["balanced"]["report"]["1"]["precision"],
            metrics["safety"]["report"]["1"]["precision"],
        ],
    })

    st.dataframe(summary_df, use_container_width=True)

    st.markdown("### Confusion Matrices")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Balanced Mode**")
        cm_bal = pd.DataFrame(
            metrics["balanced"]["confusion_matrix"],
            index=["Actual 0", "Actual 1"],
            columns=["Pred 0", "Pred 1"],
        )
        st.dataframe(cm_bal, use_container_width=True)

    with col2:
        st.markdown("**Safety Mode**")
        cm_safe = pd.DataFrame(
            metrics["safety"]["confusion_matrix"],
            index=["Actual 0", "Actual 1"],
            columns=["Pred 0", "Pred 1"],
        )
        st.dataframe(cm_safe, use_container_width=True)

# =========================================================
# TAB 5: EXPLAINABILITY
# =========================================================
with tab5:
    st.subheader("Top Predictive Words / Phrases")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top features pushing toward SuicideWatch-related**")
        pos_df = pd.DataFrame(top_positive, columns=["feature", "coefficient"])
        st.dataframe(pos_df, use_container_width=True)

    with col2:
        st.markdown("**Top features pushing toward Depression-related**")
        neg_df = pd.DataFrame(top_negative, columns=["feature", "coefficient"])
        st.dataframe(neg_df, use_container_width=True)

    st.caption(
        "These are global model signals from Logistic Regression coefficients, not per-post causal explanations."
    )

# =========================================================
# TAB 6: ETHICS
# =========================================================
with tab6:
    st.subheader("Ethical Safeguards and Responsible Use")

    st.markdown(
        """
### Intended Use
This system is designed as a research prototype for risk signal triage in social-media-like text.

### Not Intended For
- Clinical diagnosis
- Psychiatric evaluation
- Automated intervention
- Replacing human judgment

### Safeguards Included
- Dual operating modes for different risk tolerances
- Uncertainty-aware review logic
- Neither / Other guardrail for non-mental-health text
- Human-review recommendation for borderline cases
- Transparent model behavior through feature inspection
- AI-generated interpretation layer for explanation only

### Known Limitations
- Labels may overlap semantically
- Text alone does not capture full human context
- False positives and false negatives remain possible
- The Neither / Other category is currently rule-assisted, not learned from a dedicated third class
- Performance depends on the training dataset and its assumptions
- AI explanations are supportive interpretations, not ground truth
"""
    )

st.markdown("---")
st.caption(
    "Built with Streamlit, TF-IDF, Logistic Regression, a Neither/Other guardrail, "
    "browser audio recording, OpenAI speech-to-text, and AI-generated explanations."
)