from src.audio_utils import extract_features

def predict_audio(audio_path, model, scaler, pca):
    f = extract_features(audio_path).reshape(1, -1)
    f = scaler.transform(f)
    f = pca.transform(f)

    prediction = model.predict(f)[0]
    return "FAKE" if prediction == 0 else "REAL"
