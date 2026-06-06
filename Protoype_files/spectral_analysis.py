import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np


# Load an audio file
audio_path = r'WaterSound - Dataset\Cup 2\2 - 80\Recording number -  60.m4a.mp4'
y, sr = librosa.load(audio_path, sr = 48000)

print(f"Audio shape: {y.shape}")
print(f"Sample rate: {sr} Hz")
print(f"Duration: {librosa.get_duration(y=y, sr=sr):.2f} seconds")

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

print(f"MFCCs shape: {mfccs.shape}")
print(f"Number of MFCCs: {mfccs.shape[0]}")
print(f"Number of frames: {mfccs.shape[1]}")

mfccs.mean()
mfccs.std()
mfccs.max()
np.median(mfccs)


# Compute the spectral centroid
centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

print(f"Spectral Centroid shape: {centroid.shape}")
print(f"First 10 values: {centroid[0, :10]}")

# Create a figure with subplots
plt.figure(figsize=(14, 8))

# Plot the waveform
plt.subplot(3, 1, 1)
librosa.display.waveshow(y, sr=sr, alpha=0.5)
plt.title('Audio Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot the spectrogram
plt.subplot(3, 1, 2)
D = librosa.amplitude_to_db(librosa.stft(y), ref=np.max)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Log-Frequency Power Spectrogram')

# Plot the MFCCs
plt.subplot(3, 1, 3)
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.colorbar()
plt.title('MFCCs')

plt.tight_layout()
plt.show()