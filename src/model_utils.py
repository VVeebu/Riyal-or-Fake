from src.audio_utils import extract_features

def predict_audio(audio_path, model, scaler, pca):
    """
    funct to predict an unlabeled audio
    """
    f = extract_features(audio_path)
    f = f.reshape(1, -1)
    f = scaler.transform(f)
    f = pca.transform(f)
    prediction = model.predict(f)
    label = "FAKE" if prediction[0] == 0 else "REAL"
    return label
