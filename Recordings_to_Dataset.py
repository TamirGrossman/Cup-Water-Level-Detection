import os
import csv
import numpy as np
import librosa
from scipy.signal import find_peaks
import os

def extract_audio_summary(audio_path, sr=64000, n_fft=2048, hop_length=512, n_mfcc=13, cup_num = None, fill_label = None):
    """
    Analyze a single audio file and return ONE flat dictionary —
    one scalar per feature. Ready to feed directly into a classic ML model.

    For every per-window feature the following statistics are computed:
        mean, std, min, max, median, slope (linear trend in units/second)

    The slope terms are the most important for the cup-filling task:
    a rising spectral_centroid_slope or peak_frequency_slope means the
    dominant frequency is climbing — the acoustic signature of a filling cup.

    Parameters:
    -----------
    audio_path : str
        Path to the .wav file.
    sr : int
        Target sampling rate (default: 64000).
    n_fft : int
        FFT window size in samples (default: 2048).
    hop_length : int
        Hop between successive frames in samples (default: 512).
    n_mfcc : int
        Number of MFCC coefficients (default: 13).
    label : optional
        Fill level label, e.g. 'empty', 'half', 'full', or a numeric value.
        Appended as the last key if provided.

    Returns:
    --------
    dict : one flat feature vector for the whole recording.
    """

    # ── Load audio ────────────────────────────────────────────────────────────
    y, sr = librosa.load(audio_path, sr=sr)

    # ── Magnitude spectrogram (shared base) ───────────────────────────────────
    S_mag = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # ── Per-window features ───────────────────────────────────────────────────
    peak_frequency     = freqs[np.argmax(S_mag, axis=0)]
    spectral_centroid  = librosa.feature.spectral_centroid(S=S_mag, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=S_mag, sr=sr)[0]
    spectral_rolloff   = librosa.feature.spectral_rolloff(S=S_mag, sr=sr)[0]
    spectral_flatness  = librosa.feature.spectral_flatness(S=S_mag)[0]
    rms_energy         = librosa.feature.rms(S=S_mag)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(
                            y, frame_length=n_fft, hop_length=hop_length)[0]
    mfccs              = librosa.feature.mfcc(
                            y=y, sr=sr, n_mfcc=n_mfcc,
                            n_fft=n_fft, hop_length=hop_length)

    # ── Align all to the same number of windows ───────────────────────────────
    n_windows = min(
        peak_frequency.shape[0], spectral_centroid.shape[0],
        spectral_bandwidth.shape[0], spectral_rolloff.shape[0],
        spectral_flatness.shape[0], rms_energy.shape[0],
        zero_crossing_rate.shape[0], mfccs.shape[1]
    )

    named_features = {
        'spectral_centroid':  spectral_centroid[:n_windows],
        'peak_frequency':     peak_frequency[:n_windows],
        'spectral_bandwidth': spectral_bandwidth[:n_windows],
        'spectral_rolloff':   spectral_rolloff[:n_windows],
        'spectral_flatness':  spectral_flatness[:n_windows],
        'rms_energy':         rms_energy[:n_windows],
        'zero_crossing_rate': zero_crossing_rate[:n_windows],
        **{f'mfcc_{i+1}': mfccs[i, :n_windows] for i in range(n_mfcc)},
    }

    times = librosa.frames_to_time(
        np.arange(n_windows), sr=sr, hop_length=hop_length, n_fft=n_fft)

    # ── Collapse into one flat dict ───────────────────────────────────────────
    def slope(values):
        """Linear trend in units per second. Positive = rising over time."""
        if len(values) < 2:
            return 0.0
        return float(np.polyfit(times, values, 1)[0])

    summary = {
        'duration_s':    float(times[-1]),
        'Cup': cup_num if cup_num is not None else "None",
        'fill': fill_label if fill_label is not None else "None"
    }

    for name, values in named_features.items():
        summary[f'{name}_mean']   = float(np.mean(values))
        summary[f'{name}_std']    = float(np.std(values))
        summary[f'{name}_min']    = float(np.min(values))
        summary[f'{name}_max']    = float(np.max(values))
        summary[f'{name}_median'] = float(np.median(values))
        summary[f'{name}_slope']  = slope(values)   # ← key for cup-filling task

    return summary


def save_summary_csv(summaries, csv_path, append=False):
    """
    Write one or more summaries to CSV (one row per recording).

    Parameters:
    -----------
    summaries : dict or list of dict
        Output(s) of extract_audio_summary().
    csv_path : str
        Destination file path.
    append : bool
        If True and the file already exists, append without rewriting the header.
    """
    if isinstance(summaries, dict):
        summaries = [summaries]

    fieldnames = list(summaries[0].keys())
    for s in summaries[1:]:
        for k in s:
            if k not in fieldnames:
                fieldnames.append(k)

    file_has_content = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    mode         = 'a' if (append and file_has_content) else 'w'
    write_header = not (append and file_has_content)

    with open(csv_path, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for s in summaries:
            writer.writerow(s)

    print(f"Saved {len(summaries)} recording(s) → '{csv_path}'")


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Single file → one dict
    # summary = extract_audio_summary("waterBender.wav", label="half")
    # print(summary)

    # Save to CSV
    # save_summary_csv(summary, "cup_dataset.csv")
     
    rootdir = 'WaterSound - Dataset'
    # recordings = [
    #     ("waterBender.wav",  "empty"),
    # ]
    recordings = []
    for root, dirs, files in os.walk('WaterSound - Dataset'):
        for file in files:
            filepath = os.path.join(root, file)
            cup_num, fill_perc = tuple(root.split('\\')[-1].split(' - '))
            recordings.append((filepath, cup_num, fill_perc))
    # Multiple files → training table (one row per recording)
    
    dataset = [extract_audio_summary(path, cup_num=cup, fill_label=fil) for path, cup, fil in recordings]
    save_summary_csv(dataset, "cup_dataset.csv")