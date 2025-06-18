import streamlit as st

from deepface import DeepFace
import numpy as np
import time
import nltk
from nltk.stem import WordNetLemmatizer
import random
import json
import os

# --- Streamlit Page Config ---
st.set_page_config(page_title="Emotion AI Chatbot", page_icon="😊", layout="centered")

# --- Download required NLTK data ---
for data in ["punkt", "wordnet"]:
    try:
        nltk.data.find(f'tokenizers/{data}' if data == "punkt" else f'corpora/{data}')
    except LookupError:
        nltk.download(data)

# --- Initialize ---
lemmatizer = WordNetLemmatizer()
if "run_webcam" not in st.session_state:
    st.session_state.run_webcam = False
if "latest_emotion" not in st.session_state:
    st.session_state.latest_emotion = None
if "last_trigger_time" not in st.session_state:
    st.session_state.last_trigger_time = 0

# Path to the emotion JSON file (will be created in the current working directory)
emotion_json_path = "emotion.json"

def write_emotion_to_file(emotion):
    """Writes the latest emotion to a JSON file for external use."""
    try:
        with open(emotion_json_path, "w") as f:
            json.dump({"latest_emotion": emotion}, f)
    except Exception as e:
        st.error(f"Error writing emotion to file: {e}")

# --- UI Layout ---
st.title("😊 Real-time Emotion Tracker & Empathetic Chatbot")
st.write("This app detects your emotions and responds accordingly. Start your webcam to begin.")
frame_placeholder = st.empty()
col1, col2 = st.columns([1, 1])

# --- Buttons ---
if col1.button("Start Webcam"):
    st.session_state.run_webcam = True
    st.info("Webcam starting...")

if col2.button("Stop Webcam"):
    st.session_state.run_webcam = False
    st.warning("Webcam stopped.")
    frame_placeholder.empty()

# --- Chatbot Logic ---
def get_chatbot_response(emotion):
    responses = {
        "happy": ["Your smile is contagious! 😊", "What’s keeping you so joyful today?"],
        "sad": ["I’m here for you. Want to talk about it?", "Sadness is okay—let’s work through it."],
        "angry": ["Deep breaths. What made you feel this way?", "Anger is natural—let’s find calm together."],
        "fear": ["You’re safe here. Let’s face this together.", "Fear fades when shared."],
        "disgust": ["That reaction says a lot. Want to explore why?", "What’s bothering you deeply?"],
        "surprise": ["Whoa! Didn’t see that coming?", "Surprised? Tell me more!"],
        "neutral": ["Just here. And that’s okay. Let’s check in mentally."],
        "no face detected": ["Looking for your face... move closer 😊"],
        "error during analysis": ["Oops! Something went wrong. Try again."],
    }
    return random.choice(responses.get(emotion.lower(), ["I’m here to chat when you’re ready."]))

# --- Emotion Trigger UI ---
def show_emotion_ui(emotion):
    if emotion == "happy":
        st.success("😊 You look happy! Here's a quote:")
        st.markdown("> “Happiness is only real when shared.” — Into the Wild")

    elif emotion == "sad":
        st.warning("😢 Feeling low? Breathe in and try this calm video.")
        st.video("https://www.youtube.com/watch?v=inpok4MKVLM")  # 5-min meditation

    elif emotion == "angry":
        st.error("😠 Anger detected. Let's cool down.")
        st.markdown("[🌬️ Calm breathing GIF](https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.gif)")

    elif emotion == "fear":
        st.warning("😨 You seem fearful. Play some calming nature sounds.")
        st.video("https://www.youtube.com/watch?v=odvCqUguIh8")

    elif emotion == "disgust":
        st.info("😖 Something's bothering you? Try expressing it.")
        st.text_area("📝 Journal Prompt:", "What triggered this feeling today?")

    elif emotion == "surprise":
        st.balloons()
        st.success("😲 Surprise! Here's a fun fact:")
        st.markdown("> Octopuses have 3 hearts and blue blood! 🐙")

    elif emotion == "neutral":
        st.markdown("🙂 Feeling neutral. How about a check-in?")
        st.radio("🧭 How are you feeling right now?", ["Fine", "Motivated", "Tired", "Overwhelmed"], key="neutral_checkin_radio2")

# --- Webcam & Analysis Loop ---
if st.session_state.run_webcam:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam. Please check your device.")
        st.session_state.run_webcam = False

    else:
        st.success("Webcam active! Detecting emotion...")
        while st.session_state.run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Frame read failed.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_emotion = "no face detected"

            try:
                results = DeepFace.analyze(img_path=rgb_frame, actions=['emotion'], enforce_detection=False)
                if isinstance(results, list) and results:
                    result = results[0]
                    detected_emotion = result['dominant_emotion']
                    x, y, w, h = result['region']['x'], result['region']['y'], result['region']['w'], result['region']['h']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Emotion: {detected_emotion}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No face detected", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            except Exception as e:
                st.session_state.run_webcam = False
                st.error(f"DeepFace error: {e}")
                break

            # Show video frame
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

            # Emotion Triggering Every 10 Sec
            now = time.time()
            if detected_emotion and (now - st.session_state.last_trigger_time > 10):
                st.session_state.latest_emotion = detected_emotion
                write_emotion_to_file(detected_emotion)  # Write to JSON here
                show_emotion_ui(detected_emotion.lower())
                st.session_state.last_trigger_time = now

        cap.release()
        frame_placeholder.empty()
        st.info("Webcam feed ended.")

# --- Chatbot Output Section ---
st.header("🤖 Emotion-Aware Chatbot")
if st.session_state.latest_emotion:
    st.write(f"**Detected Emotion:** `{st.session_state.latest_emotion.capitalize()}`")
    st.info(f"**Chatbot says:** {get_chatbot_response(st.session_state.latest_emotion)}")
else:
    st.info("Start webcam to detect emotion and get chatbot response.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, OpenCV, DeepFace & NLTK.")
