import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import ShortTimeFFT, windows

def compute_and_display_stft(wav_file_path, nperseg=256, noverlap=None, window='hann', freq_range=None):
    """
    Compute and display the STFT of a .wav file using ShortTimeFFT.
    
    Parameters:
    -----------
    wav_file_path : str
        Path to the .wav file
    nperseg : int
        Length of each segment for STFT (default: 256). Larger values give better 
        frequency resolution but worse time resolution.
    noverlap : int or None
        Number of overlapping samples between segments (default: nperseg // 2).
        Larger overlap provides smoother results.
    window : str
        Window function to use ('hann', 'hamming', 'blackman', etc.)
    freq_range : tuple or None
        Tuple of (min_freq, max_freq) to display. If None, shows all frequencies.
    
    Returns:
    --------
    time_values : numpy.ndarray
        Time values (in seconds) for each STFT column
    frequencies : numpy.ndarray
        Frequency values (in Hz) for each STFT row
    stft_matrix : numpy.ndarray
        Complex-valued STFT matrix (magnitude can be computed as np.abs(stft_matrix))
    """
    
    # Read the .wav file
    sample_rate, audio_data = wavfile.read(wav_file_path)
    
    # If stereo, convert to mono by averaging channels
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Normalize audio to float
    if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
        audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
    
    # Set default overlap
    if noverlap is None:
        noverlap = nperseg // 2
    
    # Create window
    win = windows.get_window(window, nperseg)
    
    # Create ShortTimeFFT object
    stft_obj = ShortTimeFFT(
        win=win,
        hop=nperseg - noverlap,  # hop is the inverse of overlap
        fs=sample_rate,
        mfft=None  # Use default FFT size (next power of 2)
    )
    
    # Compute STFT
    stft_matrix = stft_obj.stft(audio_data)
    
    # Get time and frequency values
    time_values = stft_obj.t(len(audio_data))
    frequencies = stft_obj.f
    
    # Compute magnitude in dB scale
    magnitude_db = 20 * np.log10(np.abs(stft_matrix) + 1e-10)
    
    # Create the visualization
    plt.figure(1,figsize=(12, 6))
    
    # Determine frequency range for display
    if freq_range is None:
        freq_mask = slice(None)
        freqs_to_plot = frequencies
        mag_to_plot = magnitude_db
    else:
        freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        freqs_to_plot = frequencies[freq_mask]
        mag_to_plot = magnitude_db[freq_mask, :]
    
    # Plot spectrogram
    plt.pcolormesh(
        time_values,
        freqs_to_plot,
        mag_to_plot,
        shading='gouraud',
        cmap='viridis'
    )
    
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('STFT Spectrogram (ShortTimeFFT)')
    plt.colorbar(label='Magnitude (dB)')
    plt.tight_layout()
    plt.show()
    
    return time_values, frequencies, stft_matrix


def get_top_frequencies_with_plot(time_values, frequencies, stft_matrix, n_frequencies=10):
    """
    Extract and visualize the top N frequencies with the highest average power.
    
    Parameters:
    -----------
    time_values : numpy.ndarray
        Time values (in seconds) from compute_and_display_stft
    frequencies : numpy.ndarray
        Frequency values (in Hz) from compute_and_display_stft
    stft_matrix : numpy.ndarray
        Complex-valued STFT matrix from compute_and_display_stft
    n_frequencies : int
        Number of top frequencies to return (default: 10)
    
    Returns:
    --------
    top_frequencies : numpy.ndarray
        Array of the top N frequencies sorted by power (highest to lowest)
    power_values : numpy.ndarray
        Array of power values (in dB) corresponding to each top frequency
    frequency_indices : numpy.ndarray
        Indices in the original frequencies array for each top frequency
    """
    
    # Compute magnitude of the STFT matrix
    magnitude = np.abs(stft_matrix)
    
    # Compute average power across time for each frequency (in dB scale)
    avg_power_linear = np.mean(magnitude, axis=1)
    avg_power_db = 20 * np.log10(avg_power_linear + 1e-10)
    
    # Find indices of the top N frequencies
    top_indices = np.argsort(avg_power_db)[-n_frequencies:][::-1]  # Descending order
    
    # Extract top frequencies and their power values
    top_frequencies = frequencies[top_indices]
    top_power_values = avg_power_db[top_indices]
    
    # Create visualization
    fig = plt.figure(2,figsize=(12, 6))
    (ax1, ax2) = fig.subplots(1, 2)
    # Plot 1: Power spectrum with top frequencies highlighted
    ax1.semilogy(frequencies, avg_power_linear, label='Average Power Spectrum', linewidth=1.5)
    ax1.scatter(top_frequencies, 10 ** (top_power_values / 20), 
                color='red', s=100, marker='o', zorder=5, label=f'Top {n_frequencies} Frequencies')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Average Power (Linear Scale)')
    ax1.set_title('Power Spectrum with Top Frequencies Highlighted')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Bar chart of top frequencies
    rank = np.arange(1, n_frequencies + 1)
    ax2.barh(rank, top_power_values, color='steelblue')
    ax2.set_yticks(rank)
    ax2.set_yticklabels([f'{f:.1f} Hz' for f in top_frequencies])
    ax2.set_xlabel('Average Power (dB)')
    ax2.set_title(f'Top {n_frequencies} Frequencies by Power')
    ax2.invert_yaxis()
    ax2.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return top_frequencies, top_power_values, top_indices

# Example usage
if __name__ == "__main__":
    # Compute and display STFT
    time, freq, stft_result = compute_and_display_stft(
        r'Recordings\5000 Hz Tone Sound Sine Wave [OnlineSound.net].wav',
        nperseg=1024,
        window='hann',
        freq_range=(0, 10000)  # Show 0-5 kHz
    )
    
    # Access the data
    print(f"Time values shape: {time.shape}")
    print(f"Frequency values shape: {freq.shape}")
    print(f"STFT matrix shape: {stft_result.shape}")
    print(f"Time range: {time[0]:.3f} to {time[-1]:.3f} seconds")
    print(f"Frequency range: {freq[0]:.1f} to {freq[-1]:.1f} Hz")

    # Get the top 10 frequencies with visualization
    top_freqs, top_power, top_indices = get_top_frequencies_with_plot(
        time, freq, stft_result, n_frequencies=10
    )
    
    # Display results in a table
    print("\nTop 10 Frequencies by Average Power:")
    print("=" * 55)
    print(f"{'Rank':<6} {'Frequency (Hz)':<18} {'Power (dB)':<18}")
    print("-" * 55)
    for i, (freq_hz, power_db) in enumerate(zip(top_freqs, top_power), 1):
        print(f"{i:<6} {freq_hz:<18.2f} {power_db:<18.2f}")