import librosa
import numpy as np

def extract_features(path):
    """
    function to extract features from audio file
    """
    y, sr = librosa.load(path, sr=16000)

    mfcc_feat = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr), axis=1)
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr), axis=1)
    flatness = np.mean(librosa.feature.spectral_flatness(y=y), axis=1)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y), axis=1)

    return np.hstack([mfcc_feat, centroid, bandwidth, flatness, zcr])