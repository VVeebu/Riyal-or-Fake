import streamlit as st
import numpy as np
import tempfile
import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
from src.model_utils import predict_audio

# Load model assets
gb_model = joblib.load("models/gb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")

st.title("Riyal or Fake")
st.subheader("Audio Deepfake Detection")

# Create tabs
tab_pred, tab_viz = st.tabs(["🔮 Prediction", "📊 Visualization & Analysis"])

# =========================
# TAB 1 — PREDICTION
# =========================
with tab_pred:
    st.header("Audio Prediction")

    uploaded = st.file_uploader(
        "Upload WAV file for prediction",
        type=["wav"],
        key="pred_upload"
    )

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.read())
            audio_path = tmp.name

        st.audio(uploaded)

        prediction = predict_audio(audio_path, gb_model, scaler, pca)

        if prediction == "REAL":
            st.success("🟢 Prediction: REAL")
        else:
            st.error("🔴 Prediction: FAKE")

# =========================
# TAB 2 — VISUALIZATION
# =========================
with tab_viz:
    st.header("Audio Visualization & Analysis")

    uploaded_viz = st.file_uploader(
        "Upload WAV file for visualization",
        type=["wav"],
        key="viz_upload"
    )

    if uploaded_viz:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_viz.read())
            audio_path = tmp.name

        st.audio(uploaded_viz)

        # Load audio
        y, sr = librosa.load(audio_path, sr=44100)

        # Waveform
        st.subheader("Waveform")
        fig1, ax1 = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax1)
        ax1.set_title("Audio Waveform")
        st.pyplot(fig1)

        # MFCC
        st.subheader("MFCC (Mel-Frequency Cepstral Coefficients)")
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        fig2, ax2 = plt.subplots()
        img = librosa.display.specshow(
            mfcc,
            x_axis="time",
            sr=sr,
            ax=ax2
        )
        ax2.set_title("MFCC")
        fig2.colorbar(img, ax=ax2)
        st.pyplot(fig2)
        
        # Mel-Spectrogram
        st.subheader("Mel-Spectrogram")

        # Compute Mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(figsize=(12, 4))
        img = librosa.display.specshow(
            S_dB,
            sr=sr,
            x_axis="time",
            y_axis="mel",
            fmax=sr // 2,
            cmap="magma",
            ax=ax
        )

        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title("Mel-Spectrogram")
        plt.tight_layout()

        st.pyplot(fig)
        