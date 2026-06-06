import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def spectral_analysis(audio_path, sr=None, n_fft=2048, hop_length=512, n_mfcc=13):
    """
    Perform spectral analysis on an audio file and extract important frequencies.
    
    Parameters:
    -----------
    audio_path : str
        Path to the audio file
    sr : int, optional
        Sampling rate (if None, librosa will use 22050 Hz by default)
    n_fft : int
        FFT window size (default: 2048)
    hop_length : int
        Number of samples between successive frames (default: 512)
    n_mfcc : int
        Number of MFCCs to compute (default: 13)
    
    Returns:
    --------
    dict : Dictionary containing spectral features and analysis results
    """
    
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Compute Short-Time Fourier Transform (STFT)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    
    # Compute magnitude spectrum
    S_mag = np.abs(S)
    
    # Convert to dB scale
    S_db = librosa.power_to_db(S_mag**2, ref=np.max)
    
    # Create frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Average spectrum across all time frames
    avg_spectrum = np.mean(S_db, axis=1)
    
    # Extract spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    
    # Compute MFCCs (Mel-Frequency Cepstral Coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, 
                                hop_length=hop_length)
    
    # Calculate mean and std for each MFCC coefficient
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # Get top 10 MFCCs by absolute mean value
    top_n_mfcc = min(10, n_mfcc)
    mfcc_indices = np.argsort(np.abs(mfcc_mean))[-top_n_mfcc:][::-1]
    top_mfcc_mean = mfcc_mean[mfcc_indices]
    top_mfcc_std = mfcc_std[mfcc_indices]
    
    # Find peaks in the average spectrum (important frequencies)
    peaks, properties = find_peaks(avg_spectrum, height=-50, distance=5)
    
    # Get the top N peaks
    top_n = 10
    peak_indices = peaks[np.argsort(properties['peak_heights'])[-top_n:]]
    peak_indices = peak_indices[np.argsort(peak_indices)]
    
    important_freqs = freqs[peak_indices]
    peak_heights = avg_spectrum[peak_indices]
    
    # Return results
    results = {
        'sampling_rate': sr,
        'duration': len(y) / sr,
        'audio_data': y,
        'magnitude_spectrum': S_mag,
        'spectrum_db': S_db,
        'frequencies': freqs,
        'average_spectrum': avg_spectrum,
        'spectral_centroid_mean': np.mean(spectral_centroid),
        'spectral_centroid_std': np.std(spectral_centroid),
        'spectral_rolloff_mean': np.mean(spectral_rolloff),
        'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
        'zero_crossing_rate_mean': np.mean(zero_crossing_rate),
        'important_frequencies': important_freqs,
        'important_frequencies_db': peak_heights,
        'mfcc_full': mfcc,
        'mfcc_mean': mfcc_mean,
        'mfcc_std': mfcc_std,
        'top_mfcc_indices': mfcc_indices,
        'top_mfcc_mean': top_mfcc_mean,
        'top_mfcc_std': top_mfcc_std,
        'n_mfcc': n_mfcc,
        'hop_length': hop_length,
        'n_fft': n_fft
    }
    
    return results


def print_spectral_summary(results):
    """
    Print a summary of spectral analysis results including MFCCs.
    
    Parameters:
    -----------
    results : dict
        Dictionary returned from spectral_analysis()
    """
    print("=" * 70)
    print("SPECTRAL ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Sampling Rate: {results['sampling_rate']} Hz")
    print(f"Duration: {results['duration']:.2f} seconds")
    print()
    
    print("Spectral Features:")
    print(f"  - Spectral Centroid (mean): {results['spectral_centroid_mean']:.2f} Hz")
    print(f"  - Spectral Centroid (std):  {results['spectral_centroid_std']:.2f} Hz")
    print(f"  - Spectral Rolloff (mean):  {results['spectral_rolloff_mean']:.2f} Hz")
    print(f"  - Spectral Bandwidth (mean): {results['spectral_bandwidth_mean']:.2f} Hz")
    print(f"  - Zero Crossing Rate (mean): {results['zero_crossing_rate_mean']:.4f}")
    print()
    
    print("Top 10 Important Frequencies:")
    for i, (freq, db) in enumerate(zip(results['important_frequencies'], 
                                        results['important_frequencies_db']), 1):
        print(f"  {i:2d}. {freq:8.2f} Hz @ {db:6.2f} dB")
    print()
    
    print("Top 10 MFCCs (by absolute mean value):")
    print(f"{'Rank':<6} {'MFCC #':<10} {'Mean':<15} {'Std Dev':<15}")
    print("-" * 50)
    for i, idx in enumerate(results['top_mfcc_indices'], 1):
        mean_val = results['top_mfcc_mean'][i-1]
        std_val = results['top_mfcc_std'][i-1]
        print(f"{i:<6} {idx:<10} {mean_val:<15.6f} {std_val:<15.6f}")
    
    print()
    print("All MFCC Coefficients (Mean Values):")
    print(f"{'MFCC #':<10} {'Mean':<15} {'Std Dev':<15}")
    print("-" * 40)
    for i, (mean_val, std_val) in enumerate(zip(results['mfcc_mean'], results['mfcc_std'])):
        print(f"{i:<10} {mean_val:<15.6f} {std_val:<15.6f}")
    
    print("=" * 70)


def visualize_spectrum(results):
    """
    Visualize the spectral analysis results including spectrograms and MFCCs.
    
    Parameters:
    -----------
    results : dict
        Dictionary returned from spectral_analysis()
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Spectrogram
    img1 = librosa.display.specshow(
        results['spectrum_db'],
        sr=results['sampling_rate'],
        hop_length=results['hop_length'],
        x_axis='time',
        y_axis='log',
        ax=axes[0],
        cmap='magma'
    )
    axes[0].set_title('Spectrogram')
    fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')
    
    # Plot 2: Average spectrum with peaks highlighted
    freqs = results['frequencies']
    avg_spectrum = results['average_spectrum']
    important_freqs = results['important_frequencies']
    
    axes[1].plot(freqs, avg_spectrum, linewidth=1.5, label='Average Spectrum')
    axes[1].scatter(important_freqs, results['important_frequencies_db'], 
                   color='red', s=100, marker='x', linewidths=2, 
                   label='Important Frequencies')
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].set_title('Average Spectrum with Peak Detection')
    axes[1].set_xlim([0, results['sampling_rate'] / 2])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: MFCC Spectrogram
    img2 = librosa.display.specshow(
        results['mfcc_full'],
        sr=results['sampling_rate'],
        hop_length=results['hop_length'],
        x_axis='time',
        ax=axes[2],
        cmap='viridis'
    )
    axes[2].set_ylabel('MFCC Coefficient')
    axes[2].set_title(f'MFCC Spectrogram ({results["n_mfcc"]} coefficients)')
    fig.colorbar(img2, ax=axes[2], format='%+2.0f')
    
    plt.tight_layout()
    plt.show()


def plot_mfcc_statistics(results):
    """
    Create a bar plot of MFCC statistics.
    
    Parameters:
    -----------
    results : dict
        Dictionary returned from spectral_analysis()
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot all MFCC means
    axes[0].bar(range(len(results['mfcc_mean'])), results['mfcc_mean'], 
                color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('MFCC Coefficient')
    axes[0].set_ylabel('Mean Value')
    axes[0].set_title('Mean Values of All MFCC Coefficients')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot top 10 MFCCs
    top_indices = results['top_mfcc_indices']
    top_means = results['top_mfcc_mean']
    top_stds = results['top_mfcc_std']
    
    x_pos = np.arange(len(top_means))
    axes[1].bar(x_pos, top_means, yerr=top_stds, capsize=5,
                color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Rank')
    axes[1].set_ylabel('Mean Value')
    axes[1].set_title('Top 10 MFCCs (by Absolute Mean Value)')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels([f"MFCC {idx}" for idx in top_indices], rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Analyze an audio file
    audio_file = "waterBender.wav"  # Replace with your audio file
    
    results = spectral_analysis(audio_file, n_mfcc=13)
    print_spectral_summary(results)
    #visualize_spectrum(results)
    #plot_mfcc_statistics(results)
