import os
import librosa
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use the TkAgg backend for rendering
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

def load_audio_file(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    return audio, sr

def compute_mel_spectrogram(audio, sr, n_fft=512, n_mels=64):
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=n_fft)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB

def prepare_data_from_combined_folder(data_directory, n_fft, n_mels, chunk_length_sec=1, stride_sec=0.1):
    X, y = [], []
    for file_name in os.listdir(data_directory):
        if file_name.endswith('.wav'):
            try:
                speed = int(file_name.split('_')[0])  # Extract speed from file name
            except ValueError:
                continue  # Skip files with invalid names
            file_path = os.path.join(data_directory, file_name)

            # Load audio and divide into overlapping chunks
            audio, sr = load_audio_file(file_path)
            chunk_samples = int(chunk_length_sec * sr)
            stride_samples = int(stride_sec * sr)

            for start in range(0, len(audio) - chunk_samples + 1, stride_samples):
                chunk = audio[start:start + chunk_samples]
                if len(chunk) < chunk_samples:
                    continue  # Skip incomplete chunks
                S_DB = compute_mel_spectrogram(chunk, sr, n_fft, n_mels)
                X.append(S_DB)
                y.append(speed)
    return np.array(X), np.array(y)

data_directory = r"C:\Users\14432\Downloads\noise_and_ext\CableTraining\combined files"
n_fft = 512
n_mels = 64
stride_s = 0.015
chunk_length_s = 0.5

X, y = prepare_data_from_combined_folder(data_directory, n_fft, n_mels, chunk_length_sec=chunk_length_s, stride_sec=stride_s)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train[..., np.newaxis]  # Add channel dimension for CNN
X_test = X_test[..., np.newaxis]

model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2], 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")
predictions = model.predict(X_test).flatten()

# Plot predictions against true values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, s=5, alpha=0.1, color='blue')  # Smaller and more transparent dots
plt.title(f'Predicted Speed Regression (Chunk: {chunk_length_s}[sec], Stride: {stride_s}[sec]) MAE: {round(test_mae,2)}')
plt.xlabel('Actual Speeds')
plt.ylabel('Predicted Speeds')
plt.grid(True)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')  # Perfect predictions line
plt.show()

