import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import numpy as np
import os
import librosa

# load and preprocess the audio files
def load_data(data_path, duration=2):
    audio_data = []
    labels = []
    for label in os.listdir(data_path):
        class_path = os.path.join(data_path, label)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(class_path, file)
                    waveform, sr = librosa.load(file_path, sr=None, duration=duration)
                    audio_data.append(waveform)
                    labels.append(label)
    audio_data = np.array(audio_data)
    labels = np.array(labels)
    return audio_data, labels, sr

# Parameters
data_path = "D:/Elephants Documentation/Output_Audios/OverlapOutput/FinalClips(2 secs)"
input_duration = 2  # Duration of the audio clips in seconds

# Load data
X, y, sample_rate = load_data(data_path, input_duration)

# Convert labels to binary labels: "elephant" -> 1, "not elephant" -> 0
binary_labels = ['is_elephant', 'is_not_elephant']
label_mapping = {'elephant': 1, 'not_elephant': 0}

y_binary = np.array([label_mapping[label] for label in y])

# all waveforms have the same length
max_length = sample_rate * input_duration
X_padded = np.zeros((len(X), max_length))
for i in range(len(X)):
    X_padded[i, :len(X[i])] = X[i]

# Reshape data to match LSTM input requirements
X_padded = X_padded.reshape((X_padded.shape[0], max_length, 1))

# the model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(max_length, 1)),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model architecture
model.summary()

# Train
history = model.fit(X_padded, y_binary, epochs=50, batch_size=32, validation_split=0.2)

# Save
model.save('infrasound_lstm_model.h5')