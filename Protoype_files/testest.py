import csv
import os

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def spectral_analysis(audio_path, sr=None, n_fft=2048, hop_length=512):
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

    # Find peaks in the average spectrum (important frequencies)
    peaks, properties = find_peaks(avg_spectrum, height=-20, distance=5)

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
        'hop_length': hop_length,
        'n_fft': n_fft
    }

    return results


def extract_windowed_features(audio_path, sr=None, n_fft=2048, hop_length=512,
                              n_mfcc=13, center=True):
    """
    Extract per-window (per-frame) audio features for classic machine learning.

    The audio is processed in short rolling windows (STFT frames). Every feature
    is returned as a value *per window* so its evolution over time can be tracked
    (e.g. the spectral centroid / peak frequency rising as a cup fills with water).

    Features extracted per window:
        - spectral_centroid : "center of mass" of the spectrum (rises with pitch)
        - peak_frequency    : frequency bin with the highest magnitude (FFT argmax)
        - zero_crossing_rate: rate of sign changes (rises with high-freq splashing)
        - mfcc_1 .. mfcc_13 : first 13 MFCCs (timbre)
      Extra features that are useful for the same task:
        - spectral_bandwidth: spread of the spectrum around the centroid
        - spectral_rolloff  : frequency below which 85% of energy lies
        - spectral_flatness : tonal (low) vs noisy/splashy (high)
        - rms_energy        : loudness per window (splash intensity)

    Parameters:
    -----------
    audio_path : str
        Path to the audio file.
    sr : int, optional
        Target sampling rate. If None, librosa uses 22050 Hz.
    n_fft : int
        FFT window size = length of each rolling window in samples (default: 2048).
    hop_length : int
        Step between successive windows in samples (default: 512). Smaller =
        more overlap = finer time resolution.
    n_mfcc : int
        Number of MFCC coefficients to keep (default: 13).
    center : bool
        Whether to pad so frames are centered (librosa default). Kept consistent
        across every feature so all frames line up.

    Returns:
    --------
    dict with:
        'feature_matrix'   : np.ndarray, shape (n_windows, n_features)
                             Ready to feed into a classic ML model. Each row is
                             one window, each column one feature.
        'feature_names'    : list[str], column names for feature_matrix
        'times'            : np.ndarray, shape (n_windows,) time (s) of each window
        'sampling_rate'    : int
        'n_fft', 'hop_length', 'n_mfcc'
        plus each feature as its own 1-D array for convenience.
    """

    # Load audio
    y, sr = librosa.load(audio_path, sr=sr)

    # Shared magnitude spectrogram so spectral features stay consistent
    S_mag = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                                center=center))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # --- Peak Frequency Tracking ---
    # Per window: the frequency of the bin with the largest magnitude.
    peak_frequency = freqs[np.argmax(S_mag, axis=0)]

    # --- Spectral Centroid (per window) ---
    spectral_centroid = librosa.feature.spectral_centroid(S=S_mag, sr=sr)[0]

    # --- Zero-Crossing Rate (per window) ---
    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        y, frame_length=n_fft, hop_length=hop_length, center=center)[0]

    # --- MFCCs (first n_mfcc coefficients, per window) ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                 n_fft=n_fft, hop_length=hop_length)

    # --- Extra useful features (per window) ---
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S_mag, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(S=S_mag, sr=sr)[0]
    spectral_flatness = librosa.feature.spectral_flatness(S=S_mag)[0]
    rms_energy = librosa.feature.rms(S=S_mag, frame_length=n_fft,
                                     hop_length=hop_length)[0]

    # Align everything to the same number of windows (guard against off-by-one)
    n_windows = min(peak_frequency.shape[0], spectral_centroid.shape[0],
                    zero_crossing_rate.shape[0], mfccs.shape[1],
                    spectral_bandwidth.shape[0], spectral_rolloff.shape[0],
                    spectral_flatness.shape[0], rms_energy.shape[0])

    spectral_centroid = spectral_centroid[:n_windows]
    peak_frequency = peak_frequency[:n_windows]
    zero_crossing_rate = zero_crossing_rate[:n_windows]
    spectral_bandwidth = spectral_bandwidth[:n_windows]
    spectral_rolloff = spectral_rolloff[:n_windows]
    spectral_flatness = spectral_flatness[:n_windows]
    rms_energy = rms_energy[:n_windows]
    mfccs = mfccs[:, :n_windows]

    # Stack into a (n_windows, n_features) matrix
    base_features = np.vstack([
        spectral_centroid,
        peak_frequency,
        zero_crossing_rate,
        spectral_bandwidth,
        spectral_rolloff,
        spectral_flatness,
        rms_energy,
    ])
    feature_matrix = np.vstack([base_features, mfccs]).T

    feature_names = [
        'spectral_centroid',
        'peak_frequency',
        'zero_crossing_rate',
        'spectral_bandwidth',
        'spectral_rolloff',
        'spectral_flatness',
        'rms_energy',
    ] + [f'mfcc_{i + 1}' for i in range(n_mfcc)]

    times = librosa.frames_to_time(np.arange(n_windows), sr=sr,
                                   hop_length=hop_length, n_fft=n_fft)

    return {
        'feature_matrix': feature_matrix,
        'feature_names': feature_names,
        'times': times,
        'sampling_rate': sr,
        'n_fft': n_fft,
        'hop_length': hop_length,
        'n_mfcc': n_mfcc,
        'n_windows': n_windows,
        # individual series for convenience / plotting
        'spectral_centroid': spectral_centroid,
        'peak_frequency': peak_frequency,
        'zero_crossing_rate': zero_crossing_rate,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_rolloff': spectral_rolloff,
        'spectral_flatness': spectral_flatness,
        'rms_energy': rms_energy,
        'mfccs': mfccs,
    }


def aggregate_features(windowed_results, window_frames=None, agg=('mean', 'std')):
    """
    Optionally roll up the per-window features into larger summary windows.

    Useful when you want one feature vector per chunk (e.g. per second) instead
    of per STFT frame. Each big window is summarised with the requested
    statistics (mean / std / min / max).

    Parameters:
    -----------
    windowed_results : dict
        Output of extract_windowed_features().
    window_frames : int, optional
        How many consecutive STFT frames make up one summary window. If None,
        the whole clip is summarised into a single feature vector.
    agg : tuple of str
        Statistics to compute per feature. Any of: 'mean', 'std', 'min', 'max'.

    Returns:
    --------
    dict with:
        'feature_matrix' : np.ndarray, shape (n_chunks, n_features * len(agg))
        'feature_names'  : list[str]
        'times'          : np.ndarray, start time (s) of each chunk
    """
    fm = windowed_results['feature_matrix']
    names = windowed_results['feature_names']
    times = windowed_results['times']

    funcs = {
        'mean': np.mean,
        'std': np.std,
        'min': np.min,
        'max': np.max,
    }
    for a in agg:
        if a not in funcs:
            raise ValueError(f"Unknown aggregation '{a}'. Use any of {list(funcs)}.")

    if window_frames is None:
        window_frames = fm.shape[0]
    window_frames = max(1, int(window_frames))

    rows, chunk_times = [], []
    for start in range(0, fm.shape[0], window_frames):
        block = fm[start:start + window_frames]
        if block.shape[0] == 0:
            continue
        row = np.concatenate([funcs[a](block, axis=0) for a in agg])
        rows.append(row)
        chunk_times.append(times[start])

    out_names = [f'{name}_{a}' for a in agg for name in names]

    return {
        'feature_matrix': np.array(rows),
        'feature_names': out_names,
        'times': np.array(chunk_times),
    }


def save_features_csv(windowed_results, csv_path, include_time=True):
    """
    Save a per-window feature matrix to CSV (one row per window).

    Parameters:
    -----------
    windowed_results : dict
        Output of extract_windowed_features() or aggregate_features().
    csv_path : str
        Destination path.
    include_time : bool
        Prepend a 'time_s' column when times are available.
    """
    fm = windowed_results['feature_matrix']
    names = list(windowed_results['feature_names'])
    times = windowed_results.get('times')

    has_time = include_time and times is not None and len(times) == fm.shape[0]
    header = (['time_s'] if has_time else []) + names

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(fm.shape[0]):
            row = ([times[i]] if has_time else []) + list(fm[i])
            writer.writerow(row)

    print(f"Saved {fm.shape[0]} windows x {len(names)} features to '{csv_path}'")


def _trend_slope(values, times):
    """Linear trend (slope) of a per-window feature in units per second.

    Positive = feature rises over time (e.g. centroid climbing as a cup fills).
    Returns 0.0 if there aren't enough points to fit a line.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    if values.size < 2 or np.allclose(times, times[0]):
        return 0.0
    return float(np.polyfit(times, values, 1)[0])


def summarize_features(windowed_results, label=None, source_name=None, extra=None):
    """
    Collapse the per-window feature matrix into ONE flat feature vector
    (returned as a dict) that describes the whole recording.

    For every per-window feature, the following statistics are computed across
    time: mean, std, min, max, median, and slope (linear trend, units/second).
    The slope terms are what let a classic ML model "see" that, for example, the
    spectral centroid and peak frequency keep rising while a cup fills.

    The result is a plain dict (an ordered, named feature vector) -- easy to drop
    into a list and write to CSV with save_summary_csv(), one row per recording.

    Parameters:
    -----------
    windowed_results : dict
        Output of extract_windowed_features().
    label : optional
        Target/class for this recording (e.g. 'empty', 'half', 'full', or a
        numeric fill level). Written as the last column. Leave None for unlabeled
        data you intend to predict on.
    source_name : str, optional
        Identifier for the recording (e.g. the file name).
    extra : dict, optional
        Any additional metadata columns to attach (e.g. {'cup': 'mug_A'}).

    Returns:
    --------
    dict : feature vector {column_name: value}
    """
    fm = windowed_results['feature_matrix']
    names = windowed_results['feature_names']
    times = windowed_results['times']

    summary = {}
    if source_name is not None:
        summary['source'] = source_name
    summary['duration_s'] = float(times[-1]) if len(times) else 0.0
    summary['sampling_rate'] = int(windowed_results['sampling_rate'])
    summary['n_windows'] = int(fm.shape[0])
    summary['n_fft'] = int(windowed_results['n_fft'])
    summary['hop_length'] = int(windowed_results['hop_length'])

    for j, name in enumerate(names):
        col = fm[:, j]
        summary[f'{name}_mean'] = float(np.mean(col))
        summary[f'{name}_std'] = float(np.std(col))
        summary[f'{name}_min'] = float(np.min(col))
        summary[f'{name}_max'] = float(np.max(col))
        summary[f'{name}_median'] = float(np.median(col))
        summary[f'{name}_slope'] = _trend_slope(col, times)

    if extra:
        summary.update(extra)
    if label is not None:
        summary['label'] = label

    return summary


def save_summary_csv(summaries, csv_path, append=False):
    """
    Write one or more clip-level feature vectors to CSV (one row per recording).

    Parameters:
    -----------
    summaries : dict or list of dict
        A single summary from summarize_features(), or a list of them.
    csv_path : str
        Destination path.
    append : bool
        If True and the file already has content, append rows without rewriting
        the header (handy for building a dataset incrementally across files).
    """
    if isinstance(summaries, dict):
        summaries = [summaries]
    if not summaries:
        raise ValueError("No summaries to write.")

    # Union of keys, preserving first-seen order, so rows with slightly
    # different metadata still line up.
    fieldnames = list(summaries[0].keys())
    for s in summaries[1:]:
        for k in s:
            if k not in fieldnames:
                fieldnames.append(k)

    file_has_content = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    mode = 'a' if (append and file_has_content) else 'w'
    write_header = not (append and file_has_content)

    with open(csv_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for s in summaries:
            writer.writerow(s)

    print(f"Saved {len(summaries)} recording(s) x {len(fieldnames)} columns "
          f"to '{csv_path}'")


def build_feature_dataset(audio_items, csv_path, sr=None, n_fft=2048,
                          hop_length=512, n_mfcc=13):
    """
    Turn a set of recordings into a single classic-ML training table (CSV),
    one row per recording. This is the recommended end-to-end entry point.

    Parameters:
    -----------
    audio_items :
        Any of:
          - list of file paths              -> ['a.wav', 'b.wav']      (no labels)
          - list of (path, label) tuples    -> [('a.wav', 'empty'), ...]
          - dict {path: label}              -> {'a.wav': 'empty', ...}
    csv_path : str
        Where to write the dataset.
    sr, n_fft, hop_length, n_mfcc :
        Passed straight through to extract_windowed_features().

    Returns:
    --------
    list of dict : the feature vectors that were written (also useful in-memory).
    """
    # Normalise input into a list of (path, label) pairs
    if isinstance(audio_items, dict):
        items = list(audio_items.items())
    else:
        items = []
        for entry in audio_items:
            if isinstance(entry, (tuple, list)):
                items.append((entry[0], entry[1] if len(entry) > 1 else None))
            else:
                items.append((entry, None))

    summaries = []
    for path, label in items:
        windowed = extract_windowed_features(
            path, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)
        summaries.append(summarize_features(
            windowed, label=label, source_name=os.path.basename(path)))

    save_summary_csv(summaries, csv_path)
    return summaries


def print_spectral_summary(results):
    """
    Print a summary of spectral analysis results.

    Parameters:
    -----------
    results : dict
        Dictionary returned from spectral_analysis()
    """
    print("=" * 60)
    print("SPECTRAL ANALYSIS SUMMARY")
    print("=" * 60)
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
    print("=" * 60)


def print_windowed_summary(windowed_results):
    """
    Print a quick summary of the per-window feature matrix.

    Parameters:
    -----------
    windowed_results : dict
        Dictionary returned from extract_windowed_features().
    """
    fm = windowed_results['feature_matrix']
    names = windowed_results['feature_names']
    print("=" * 60)
    print("WINDOWED FEATURE EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Sampling Rate : {windowed_results['sampling_rate']} Hz")
    print(f"Window size   : {windowed_results['n_fft']} samples "
          f"(~{1000 * windowed_results['n_fft'] / windowed_results['sampling_rate']:.1f} ms)")
    print(f"Hop length    : {windowed_results['hop_length']} samples")
    print(f"Feature matrix: {fm.shape[0]} windows x {fm.shape[1]} features")
    print()
    print("Per-feature stats (mean +/- std over time):")
    for j, name in enumerate(names):
        col = fm[:, j]
        print(f"  - {name:20s}: {np.mean(col):10.3f} +/- {np.std(col):8.3f}")
    print("=" * 60)


def visualize_spectrum(results):
    """
    Visualize the spectral analysis results.

    Parameters:
    -----------
    results : dict
        Dictionary returned from spectral_analysis()
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Spectrogram
    img = librosa.display.specshow(
        results['spectrum_db'],
        sr=results['sampling_rate'],
        hop_length=results['hop_length'],
        x_axis='time',
        y_axis='log',
        ax=axes[0],
        cmap='magma'
    )
    axes[0].set_title('Spectrogram')
    fig.colorbar(img, ax=axes[0], format='%+2.0f dB')

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

    plt.tight_layout()
    plt.show()


def visualize_windowed_features(windowed_results):
    """
    Plot how the key features evolve over time (great for spotting the
    pitch/centroid rise and the increase in splashing as a cup fills).

    Parameters:
    -----------
    windowed_results : dict
        Dictionary returned from extract_windowed_features().
    """
    t = windowed_results['times']
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(t, windowed_results['spectral_centroid'], color='tab:blue')
    axes[0].set_ylabel('Hz')
    axes[0].set_title('Spectral Centroid over time')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, windowed_results['peak_frequency'], color='tab:orange')
    axes[1].set_ylabel('Hz')
    axes[1].set_title('Peak Frequency over time')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, windowed_results['zero_crossing_rate'], color='tab:green')
    axes[2].set_ylabel('rate')
    axes[2].set_title('Zero-Crossing Rate over time')
    axes[2].grid(True, alpha=0.3)

    img = librosa.display.specshow(
        windowed_results['mfccs'],
        sr=windowed_results['sampling_rate'],
        hop_length=windowed_results['hop_length'],
        x_axis='time',
        ax=axes[3],
        cmap='viridis'
    )
    axes[3].set_title(f"MFCCs (first {windowed_results['n_mfcc']})")
    axes[3].set_ylabel('MFCC index')
    fig.colorbar(img, ax=axes[3])

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Analyze an audio file
    audio_file = "waterBender.wav"  # Replace with your audio file

    # --- Global spectral summary (original behaviour) ---
    results = spectral_analysis(audio_file, sr=64000)
    print_spectral_summary(results)
    visualize_spectrum(results)

    # --- Per-window features for classic ML ---
    windowed = extract_windowed_features(audio_file, sr=64000,
                                         n_fft=2048, hop_length=512, n_mfcc=13)
    print_windowed_summary(windowed)

    # Frame-level CSV: one row PER WINDOW (good for sequence/frame labels)
    save_features_csv(windowed, "waterBender_features.csv")

    # Clip-level CSV: ONE feature vector for the whole recording, with trend
    # terms. This is the standard classic-ML table -- one row per recording.
    summary = summarize_features(windowed, label="full",
                                 source_name=audio_file)
    save_summary_csv(summary, "waterBender_summary.csv")

    visualize_windowed_features(windowed)

    # Build a training table from many recordings in one call, e.g.:
    # build_feature_dataset(
    #     [("empty.wav", "empty"), ("half.wav", "half"), ("full.wav", "full")],
    #     csv_path="cup_dataset.csv", sr=64000)

    # X is ready for scikit-learn: shape (n_windows, n_features)
    X = windowed['feature_matrix']
    feature_names = windowed['feature_names']