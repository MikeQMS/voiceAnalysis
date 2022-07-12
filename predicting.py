import librosa
import soundfile
import pickle
import numpy as np


def extract_feature(file_name, rate, mfcc, chroma, mel):
    if not isinstance(file_name, bytes):
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate
            if chroma:
                stft = np.abs(librosa.stft(X, n_fft=512))
            result = np.array([])
            if mfcc:
                mfccs = np.mean(librosa.feature.mfcc(
                    y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result = np.hstack((result, mfccs))
            if chroma:
                chroma = np.mean(librosa.feature.chroma_stft(
                    S=stft, sr=sample_rate).T, axis=0)
                result = np.hstack((result, chroma))
            if mel:
                mel = np.mean(librosa.feature.melspectrogram(
                    y=X, sr=sample_rate).T, axis=0)
                result = np.hstack((result, mel))
        return result
    else:
        X = np.frombuffer(file_name, dtype=np.float32)
        sample_rate = rate
        if chroma:
            stft = np.abs(librosa.stft(X, n_fft=512))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(
                S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(
                y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        return result


def load_feature(filename ='test1.wav', rate=None):
    x = []
    feature = extract_feature(filename, rate,  mfcc=True, chroma=True, mel=True)
    x.append(feature)
    return x


def predicting(file_name, rate=None):
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    X = load_feature(file_name, rate)
    prediction = loaded_model.predict(X)
    print("Prediction: ", prediction)
    return prediction


if __name__ == "__main__":
    for i in range(10): # the amount, how often a prediction for the file should be made (10 times)
        predicting("test2.wav") # Change file path / name to do .wav file prediction and run predicting.py
